package simulation.core;

import core.IEnvironmentState;
import rl.ReinforcementLearningFramework;
import rl.ReinforcementLearningFramework.Action;
import rl.ReinforcementLearningFramework.State;
import rl.ReinforcementLearningFramework.StepResult;
import simulation.core.NetworkTopology.NetworkNode;

import java.util.*;

/**
 * Bridge between NetworkSimulation and the reinforcement-learning environment
 * interface. Provides six scenario profiles (DDOS, INTRUSION, RANSOMWARE,
 * PORT_SCAN, DATA_EXFILTRATION, MIXED), each with distinct attack rates,
 * initial conditions, and dynamics. Call setScenarioProfile(name) before
 * reset() to select a scenario; defaults to DDOS if unset.
 */
public class NetworkSimulationEnvironment implements ReinforcementLearningFramework.Environment {

    private final NetworkSimulation simulation;
    private final NetworkEnvironment networkEnv;
    private final List<String> actionableNodes;
    private final Random random;
    private State currentState;

    // Action types available to RL agent
    public enum ActionType {
        DEFEND_NODE(0),
        ISOLATE_NODE(1),
        PATCH_NODE(2),
        MONITOR_NODE(3),
        RESTORE_NODE(4),
        DO_NOTHING(5);

        private final int id;
        ActionType(int id) { this.id = id; }
        public int getId() { return id; }
    }

    // Configuration
    private static final int FEATURES_PER_NODE = 4;
    private int stateSize = 100;
    private int actionSize;

    // ==================== Attack Tracking ====================
    private int previousCompromised;
    private int previousDefended;
    private int stepsWithoutProgress;
    private int totalSteps;
    private double cumulativeDamage;
    private Set<String> everCompromised;
    private Set<String> successfulDefenses;

    // ==================== Defense Decay Tracking ====================
    private Map<String, Integer> defenseAge;

    // ==================== Scenario Profile ====================
    /**
     * Scenario-specific attack parameters.
     *
     * Each scenario creates a genuinely different challenge:
     * - DDOS: Fast overwhelming burst attacks, many initial threats
     * - INTRUSION: Slow creeping infiltration, hard to maintain defense
     * - RANSOMWARE: Explosive cascade through vulnerable nodes
     * - PORT_SCAN: Constant reconnaissance, rapid vulnerability emergence
     * - DATA_EXFILTRATION: Targeted attacks on high-connectivity nodes
     * - MIXED: Alternating attack intensity mid-episode
     */
    private static class ScenarioProfile {
        final String name;
        final double spreadRateVulnerable;  // Spread chance to vulnerable neighbor
        final double spreadRateClean;       // Spread chance to clean neighbor
        final double externalAttackRate;    // Per-step chance for external attack on vuln node
        final double defenseDecayRate;      // Per-step chance of losing defense
        final double newVulnRate;           // Per-step chance of new vulnerability event
        final int newVulnMaxNodes;          // Max nodes affected per vuln event
        final int initialCompromised;       // Nodes compromised at episode start
        final double initialVulnFraction;   // Fraction of remaining nodes with vulnerabilities
        final double terminalFraction;      // Fraction compromised = game over

        ScenarioProfile(String name, double spreadVuln, double spreadClean, double external,
                        double decay, double newVuln, int vulnMax, int initComp,
                        double initVulnFrac, double terminal) {
            this.name = name;
            this.spreadRateVulnerable = spreadVuln;
            this.spreadRateClean = spreadClean;
            this.externalAttackRate = external;
            this.defenseDecayRate = decay;
            this.newVulnRate = newVuln;
            this.newVulnMaxNodes = vulnMax;
            this.initialCompromised = initComp;
            this.initialVulnFraction = initVulnFrac;
            this.terminalFraction = terminal;
        }
    }

    // Predefined scenario profiles
    private static final Map<String, ScenarioProfile> PROFILES = new LinkedHashMap<>();
    static {
        //                          name             spVuln spCln  ext    decay  nVuln vMax init vulnF  term
        PROFILES.put("DDOS",
                new ScenarioProfile("DDOS",              0.10,  0.03,  0.04,  0.02,  0.04, 1,   2,   0.40,  0.60));
        PROFILES.put("INTRUSION",
                new ScenarioProfile("INTRUSION",         0.06,  0.02,  0.01,  0.05,  0.08, 2,   1,   0.50,  0.60));
        PROFILES.put("RANSOMWARE",
                new ScenarioProfile("RANSOMWARE",        0.15,  0.04,  0.01,  0.03,  0.02, 1,   1,   0.20,  0.55));
        PROFILES.put("PORT_SCAN",
                new ScenarioProfile("PORT_SCAN",         0.04,  0.01,  0.06,  0.04,  0.12, 2,   0,   0.60,  0.60));
        PROFILES.put("DATA_EXFILTRATION",
                new ScenarioProfile("DATA_EXFILTRATION", 0.12,  0.03,  0.02,  0.03,  0.04, 1,   1,   0.30,  0.60));
        PROFILES.put("MIXED",
                new ScenarioProfile("MIXED",             0.08,  0.02,  0.03,  0.03,  0.06, 1,   1,   0.35,  0.60));
    }

    /** Active scenario profile (set via setScenarioProfile before reset) */
    private ScenarioProfile activeProfile;

    // ==================== Reward Configuration ====================
    private static final double REWARD_DEFEND_COMPROMISED = 10.0;
    private static final double REWARD_DEFEND_VULNERABLE = 3.0;
    private static final double REWARD_DEFEND_SAFE = -2.0;
    private static final double REWARD_DEFEND_ALREADY = -1.0;
    private static final double REWARD_ISOLATE_COMPROMISED = 7.0;
    private static final double REWARD_ISOLATE_CLEAN = -3.0;
    private static final double REWARD_PATCH_VULNERABLE = 5.0;
    private static final double REWARD_PATCH_CLEAN = -1.0;
    private static final double REWARD_MONITOR_SUSPICIOUS = 1.0;
    private static final double REWARD_MONITOR_SAFE = 0.0;
    private static final double REWARD_RESTORE_NEEDED = 3.0;
    private static final double REWARD_RESTORE_UNNEEDED = -1.0;
    private static final double REWARD_DO_NOTHING = -2.0;

    private static final double PENALTY_NEW_COMPROMISE = -8.0;
    private static final double REWARD_CLEANED_NODE = 3.0;
    private static final double TIME_PENALTY = -0.3;

    private static final double TERMINAL_PENALTY = -30.0;
    private static final double SURVIVAL_BONUS_MAX = 30.0;

    public NetworkSimulationEnvironment() {
        this(new NetworkSimulation());
    }

    public NetworkSimulationEnvironment(NetworkSimulation simulation) {
        this.simulation = simulation;
        this.networkEnv = new NetworkEnvironment(simulation.getTopology());
        this.actionableNodes = new ArrayList<>();
        this.random = new Random();
        this.everCompromised = new HashSet<>();
        this.successfulDefenses = new HashSet<>();
        this.defenseAge = new HashMap<>();
        this.activeProfile = PROFILES.get("DDOS");  // Default

        if (simulation.getTopology().getNodeCount() == 0) {
            simulation.getTopology().generateTopology(2, 5, 1);
        }

        initializeActionSpace();
    }

    /**
     * Set the scenario profile BEFORE calling reset().
     *
     * Each scenario creates genuinely different attack dynamics.
     * Supported: DDOS, INTRUSION, RANSOMWARE, PORT_SCAN, DATA_EXFILTRATION, MIXED.
     *
     * @param scenarioName Name of the scenario (case-insensitive, underscores optional)
     */
    public void setScenarioProfile(String scenarioName) {
        String key = scenarioName.toUpperCase().replace(" ", "_");
        ScenarioProfile profile = PROFILES.get(key);
        if (profile != null) {
            this.activeProfile = profile;
        } else {
            System.err.println("[Environment] Unknown scenario: " + scenarioName + ", using DDOS");
            this.activeProfile = PROFILES.get("DDOS");
        }
    }

    /** Get list of available scenario names */
    public static Set<String> getAvailableScenarios() {
        return PROFILES.keySet();
    }

    private void initializeActionSpace() {
        actionableNodes.clear();
        for (NetworkNode node : simulation.getTopology().getNodes()) {
            actionableNodes.add(node.getId());
        }
        actionSize = ActionType.values().length * Math.max(1, actionableNodes.size());
    }

    @Override
    public State reset() {
        simulation.reset();
        networkEnv.reset();
        initializeActionSpace();

        previousCompromised = 0;
        previousDefended = 0;
        stepsWithoutProgress = 0;
        totalSteps = 0;
        cumulativeDamage = 0;
        everCompromised.clear();
        successfulDefenses.clear();
        defenseAge.clear();

        // Use scenario-specific initial conditions
        setupInitialThreat();

        previousCompromised = countCompromisedNodes();
        previousDefended = countDefendedNodes();

        currentState = encodeState(buildEnvState());
        return currentState;
    }

    /**
     * Set up initial threat using active scenario profile.
     * Different scenarios start with different numbers of compromised/vulnerable nodes.
     */
    private void setupInitialThreat() {
        List<NetworkNode> nodes = new ArrayList<>(simulation.getTopology().getNodes());
        int totalNodes = nodes.size();
        Collections.shuffle(nodes, random);

        ScenarioProfile p = activeProfile;

        // Phase 1: Compromise initial nodes
        int compromiseCount = Math.max(0, Math.min(p.initialCompromised, totalNodes - 1));
        for (int i = 0; i < compromiseCount; i++) {
            NetworkNode node = nodes.get(i);
            node.setCompromised(true);
            node.setDefended(false);
            if (node.getVulnerabilities().isEmpty()) {
                node.getVulnerabilities().add("CVE-INIT-" + random.nextInt(9999));
            }
            everCompromised.add(node.getId());
            cumulativeDamage += 10;
        }

        // Phase 2: Add vulnerabilities to fraction of remaining nodes
        int vulnCount = Math.max(0, (int) Math.round((totalNodes - compromiseCount) * p.initialVulnFraction));
        for (int i = compromiseCount; i < Math.min(compromiseCount + vulnCount, totalNodes); i++) {
            NetworkNode node = nodes.get(i);
            if (!node.isCompromised() && node.getVulnerabilities().isEmpty()) {
                node.getVulnerabilities().add("CVE-2024-" + random.nextInt(9999));
            }
        }

        // PORT_SCAN special: if 0 initial compromised, an external attack appears
        // around step ~3 (handled naturally by the high external attack rate).
    }

    @Override
    public StepResult step(Action action) {
        totalSteps++;

        // Decode action
        int actionIndex = action.getId();
        int totalActionTypes = ActionType.values().length;
        int actionTypeId = actionIndex / Math.max(1, actionableNodes.size());
        int nodeIndex = actionIndex % Math.max(1, actionableNodes.size());

        if (actionTypeId >= totalActionTypes) {
            actionTypeId = ActionType.DO_NOTHING.getId();
        }
        if (nodeIndex >= actionableNodes.size()) {
            nodeIndex = 0;
        }

        ActionType actionType = ActionType.values()[actionTypeId];
        String nodeId = actionableNodes.isEmpty() ? null : actionableNodes.get(nodeIndex);

        NetworkNode targetNode = nodeId != null ? simulation.getTopology().getNodeMap().get(nodeId) : null;
        boolean wasCompromised = targetNode != null && targetNode.isCompromised();
        boolean wasDefended = targetNode != null && targetNode.isDefended();
        boolean hadVulnerabilities = targetNode != null && !targetNode.getVulnerabilities().isEmpty();

        double actionReward = executeActionWithReward(actionType, nodeId, targetNode,
                wasCompromised, wasDefended, hadVulnerabilities);

        simulation.step();
        simulateAttackProgression();
        applyDefenseDecay();

        IEnvironmentState simState = buildEnvState();
        State nextState = encodeState(simState);
        currentState = nextState;

        double stateReward = calculateStateChangeReward(simState);

        boolean done = isTerminal(simState);

        double episodeEndReward = 0;
        if (done) {
            int currentCompromised = ((Number) simState.getStateVariables()
                    .getOrDefault("compromisedNodes", 0)).intValue();
            int totalNodes = simulation.getTopology().getNodeCount();
            double compromiseFraction = (double) currentCompromised / Math.max(1, totalNodes);

            if (compromiseFraction >= activeProfile.terminalFraction) {
                episodeEndReward = TERMINAL_PENALTY;
            } else {
                double healthFraction = 1.0 - compromiseFraction;
                episodeEndReward = SURVIVAL_BONUS_MAX * healthFraction;
            }
        }

        double reward = actionReward + stateReward + episodeEndReward;

        StepResult result = new StepResult(nextState, reward, done);
        result.info.put("action", actionType.toString());
        result.info.put("node", nodeId);
        result.info.put("actionReward", actionReward);
        result.info.put("stateReward", stateReward);
        result.info.put("episodeEndReward", episodeEndReward);
        result.info.put("compromisedNodes", simState.getStateVariables().get("compromisedNodes"));
        result.info.put("defendedNodes", simState.getStateVariables().get("defendedNodes"));

        return result;
    }

    /**
     * Execute action and return immediate action-specific reward.
     */
    private double executeActionWithReward(ActionType actionType, String nodeId, NetworkNode node,
                                           boolean wasCompromised, boolean wasDefended, boolean hadVulnerabilities) {
        if (node == null) {
            return -1.0;
        }

        double reward = 0;

        switch (actionType) {
            case DEFEND_NODE:
                if (wasCompromised) {
                    node.setCompromised(false);
                    node.setDefended(true);
                    defenseAge.put(nodeId, 0);
                    successfulDefenses.add(nodeId);
                    reward = REWARD_DEFEND_COMPROMISED;
                } else if (hadVulnerabilities) {
                    node.setDefended(true);
                    defenseAge.put(nodeId, 0);
                    reward = REWARD_DEFEND_VULNERABLE;
                } else if (!wasDefended) {
                    node.setDefended(true);
                    defenseAge.put(nodeId, 0);
                    reward = REWARD_DEFEND_SAFE;
                } else {
                    defenseAge.put(nodeId, 0);
                    reward = REWARD_DEFEND_ALREADY;
                }
                break;

            case ISOLATE_NODE:
                if (wasCompromised) {
                    node.setDefended(true);
                    defenseAge.put(nodeId, 0);
                    networkEnv.updateState("isolated_" + nodeId, true);
                    reward = REWARD_ISOLATE_COMPROMISED;
                } else {
                    networkEnv.updateState("isolated_" + nodeId, true);
                    reward = REWARD_ISOLATE_CLEAN;
                }
                break;

            case PATCH_NODE:
                if (hadVulnerabilities) {
                    node.getVulnerabilities().clear();
                    node.setDefended(true);
                    defenseAge.put(nodeId, 0);
                    reward = REWARD_PATCH_VULNERABLE;
                } else {
                    reward = REWARD_PATCH_CLEAN;
                }
                break;

            case MONITOR_NODE:
                networkEnv.updateState("monitoring_" + nodeId, true);
                if (hadVulnerabilities || wasCompromised) {
                    reward = REWARD_MONITOR_SUSPICIOUS;
                } else {
                    reward = REWARD_MONITOR_SAFE;
                }
                break;

            case RESTORE_NODE:
                Boolean isIsolated = (Boolean) networkEnv.getState().get("isolated_" + nodeId);
                if (isIsolated != null && isIsolated) {
                    networkEnv.updateState("isolated_" + nodeId, false);
                    reward = REWARD_RESTORE_NEEDED;
                } else {
                    reward = REWARD_RESTORE_UNNEEDED;
                }
                break;

            case DO_NOTHING:
                reward = REWARD_DO_NOTHING;
                break;
        }

        return reward;
    }

    /**
     * Simulate attack progression using active scenario profile.
     * Attack rates come from the scenario profile, not static constants.
     */
    private void simulateAttackProgression() {
        NetworkTopology topo = simulation.getTopology();
        ScenarioProfile p = activeProfile;
        List<NetworkNode> compromised = new ArrayList<>();

        for (NetworkNode node : topo.getNodes()) {
            if (node.isCompromised()) {
                compromised.add(node);
                everCompromised.add(node.getId());
            }
        }

        // Vector 1: Spread from compromised to neighbors (scenario-specific rates)
        for (NetworkNode infected : compromised) {
            Boolean isIsolated = (Boolean) networkEnv.getState().get("isolated_" + infected.getId());
            if (isIsolated != null && isIsolated) continue;

            for (NetworkNode neighbor : topo.getNeighbors(infected)) {
                if (!neighbor.isCompromised() && !neighbor.isDefended()) {
                    double spreadChance = neighbor.getVulnerabilities().isEmpty()
                            ? p.spreadRateClean : p.spreadRateVulnerable;

                    // MIXED scenario: alternate attack intensity every 50 steps
                    if ("MIXED".equals(p.name) && (totalSteps / 50) % 2 == 1) {
                        spreadChance *= 1.5;  // Burst phase
                    }

                    if (random.nextDouble() < spreadChance) {
                        neighbor.setCompromised(true);
                        cumulativeDamage += 10;
                    }
                }
            }
        }

        // Vector 2: External attacks on vulnerable undefended nodes
        for (NetworkNode node : topo.getNodes()) {
            if (!node.isCompromised() && !node.isDefended() && !node.getVulnerabilities().isEmpty()) {
                if (random.nextDouble() < p.externalAttackRate) {
                    node.setCompromised(true);
                    cumulativeDamage += 15;
                }
            }
        }

        // Vector 3: New vulnerabilities emerging
        if (random.nextDouble() < p.newVulnRate) {
            List<NetworkNode> nodes = new ArrayList<>(topo.getNodes());
            Collections.shuffle(nodes, random);
            int added = 0;
            for (NetworkNode node : nodes) {
                if (added >= p.newVulnMaxNodes) break;
                if (!node.isDefended() && !node.isCompromised() && node.getVulnerabilities().size() < 3) {
                    node.getVulnerabilities().add("CVE-NEW-" + random.nextInt(9999));
                    added++;
                }
            }
        }
    }

    /**
     * Apply defense decay using scenario-specific rate.
     */
    private void applyDefenseDecay() {
        NetworkTopology topo = simulation.getTopology();

        for (NetworkNode node : topo.getNodes()) {
            if (node.isDefended()) {
                String nodeId = node.getId();
                int age = defenseAge.getOrDefault(nodeId, 0) + 1;
                defenseAge.put(nodeId, age);

                if (random.nextDouble() < activeProfile.defenseDecayRate) {
                    node.setDefended(false);
                    defenseAge.remove(nodeId);
                }
            }
        }
    }

    /**
     * Calculate reward based on state changes.
     */
    private double calculateStateChangeReward(IEnvironmentState state) {
        Map<String, Object> stateVars = state.getStateVariables();
        int currentCompromised = ((Number) stateVars.getOrDefault("compromisedNodes", 0)).intValue();
        int currentDefended = ((Number) stateVars.getOrDefault("defendedNodes", 0)).intValue();

        double reward = 0;

        int newCompromises = currentCompromised - previousCompromised;
        if (newCompromises > 0) {
            reward += newCompromises * PENALTY_NEW_COMPROMISE;
            stepsWithoutProgress = 0;
        }

        int cleaned = previousCompromised - currentCompromised;
        if (cleaned > 0) {
            reward += cleaned * REWARD_CLEANED_NODE;
            stepsWithoutProgress = 0;
        }

        reward += TIME_PENALTY;

        if (newCompromises == 0 && cleaned == 0) {
            stepsWithoutProgress++;
        }

        previousCompromised = currentCompromised;
        previousDefended = currentDefended;

        return reward;
    }

    private int countCompromisedNodes() {
        int count = 0;
        for (NetworkNode node : simulation.getTopology().getNodes()) {
            if (node.isCompromised()) count++;
        }
        return count;
    }

    private int countDefendedNodes() {
        int count = 0;
        for (NetworkNode node : simulation.getTopology().getNodes()) {
            if (node.isDefended()) count++;
        }
        return count;
    }

    private int countVulnerableNodes() {
        int count = 0;
        for (NetworkNode node : simulation.getTopology().getNodes()) {
            if (!node.getVulnerabilities().isEmpty() && !node.isDefended()) count++;
        }
        return count;
    }

    @Override
    public int getStateSize() {
        return stateSize;
    }

    @Override
    public int getActionSize() {
        return actionSize;
    }

    @Override
    public State getCurrentState() {
        if (currentState == null) {
            currentState = encodeState(buildEnvState());
        }
        return currentState;
    }

    /**
     * Encode IEnvironmentState into RL State with fixed-size feature vector.
     */
    private State encodeState(IEnvironmentState simState) {
        Map<String, Object> stateVars = simState.getStateVariables();
        double[] features = new double[stateSize];
        int idx = 0;

        int totalNodes = simulation.getTopology().getNodeCount();
        int compromisedCount = ((Number) stateVars.getOrDefault("compromisedNodes", 0)).intValue();
        int defendedCount = ((Number) stateVars.getOrDefault("defendedNodes", 0)).intValue();

        if (idx < stateSize) features[idx++] = (double) compromisedCount / Math.max(1, totalNodes);
        if (idx < stateSize) features[idx++] = (double) defendedCount / Math.max(1, totalNodes);
        if (idx < stateSize) features[idx++] = (double) simState.getTimestamp() / 300.0;
        if (idx < stateSize) features[idx++] = (double) countVulnerableNodes() / Math.max(1, totalNodes);
        if (idx < stateSize) features[idx++] = Math.min(1.0, cumulativeDamage / 1000.0);

        for (NetworkNode node : simulation.getTopology().getNodes()) {
            if (idx + FEATURES_PER_NODE > stateSize) break;

            features[idx++] = node.isCompromised() ? 1.0 : 0.0;
            features[idx++] = node.isDefended() ? 1.0 : 0.0;
            features[idx++] = (double) node.getVulnerabilities().size() / 5.0;

            long connections = simulation.getTopology().getConnections().stream()
                    .filter(c -> c.getSource().equals(node) || c.getTarget().equals(node))
                    .count();
            features[idx++] = (double) connections / 10.0;
        }

        while (idx < stateSize) {
            features[idx++] = 0.0;
        }

        return new State(features);
    }

    /**
     * Check if simulation has reached terminal state.
     * Uses scenario-specific terminal threshold.
     */
    private boolean isTerminal(IEnvironmentState state) {
        Map<String, Object> stateVars = state.getStateVariables();
        int compromised = ((Number) stateVars.getOrDefault("compromisedNodes", 0)).intValue();
        int totalNodes = simulation.getTopology().getNodeCount();

        if (totalNodes > 0 && compromised >= Math.ceil(totalNodes * activeProfile.terminalFraction)) {
            return true;
        }

        if (state.getTimestamp() > 500) {
            return true;
        }

        return false;
    }

    public NetworkTopology getTopology() {
        return simulation.getTopology();
    }

    public NetworkSimulation getSimulation() {
        return simulation;
    }

    public void setTopologySize(int servers, int workstations, int routers) {
        simulation.getTopology().generateTopology(servers, workstations, routers);
        initializeActionSpace();
        reset();
    }

    // Build environment state from simulation
    private static final class SimpleEnvState implements IEnvironmentState {
        private final Map<String, Object> vars;
        private final long ts;
        SimpleEnvState(Map<String, Object> vars, long ts) {
            this.vars = vars; this.ts = ts;
        }
        @Override public Map<String, Object> getStateVariables() { return vars; }
        @Override public Object getVariable(String name) { return vars.get(name);}
        @Override public long getTimestamp() { return ts; }
    }

    private IEnvironmentState buildEnvState() {
        NetworkTopology topo = simulation.getTopology();
        int compromised = (int) topo.getNodes().stream().filter(NetworkNode::isCompromised).count();
        int defended = (int) topo.getNodes().stream().filter(NetworkNode::isDefended).count();
        Map<String, Object> m = new HashMap<>();
        m.put("compromisedNodes", compromised);
        m.put("defendedNodes", defended);
        long ts = simulation.getTimeStep();
        return new SimpleEnvState(m, ts);
    }
}