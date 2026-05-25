package simulation.core;

import simulation.agents.AttackerAgent;
import simulation.agents.DefenderAgent;
import simulation.agents.SimulationAgent;
import simulation.attacks.*;
import simulation.events.AlertEvent;
import simulation.events.DefenderActionEvent;
import visualization.simulation.recording.AgentStateSnapshot;
import visualization.simulation.recording.AttackSnapshot;
import visualization.simulation.recording.NetworkStateSnapshot;
import visualization.simulation.recording.NodeSnapshot;

import java.awt.*;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Core network simulation engine for the cybersecurity scenarios used in Experiment 2.
 * Maintains the network state across time steps and applies the per-step update
 * mechanics described in Section 3.4 of the accompanying paper.
 */
public class NetworkSimulation {
    private NetworkTopology topology;
    private List<SimulationAgent> agents;
    private Map<String, AttackScenario> activeAttacks;
    private List<SimulationEvent> eventLog;
    private Map<String, List<Alert>> alerts;
    private int timeStep;
    private boolean isRunning;

    // Traffic flow visualization data
    private Map<String, TrafficFlow> trafficFlows;

    // ----------------------------------------------------------------
    //  Instrumentation logs.
    //  These back the case-study metrics M8 (total defender actions in
    //  an interval) and M9 (first detection latency). They are additive
    //  and do not affect any simulation dynamics — defender behaviour is
    //  untouched beyond emitting events.
    // ----------------------------------------------------------------
    private List<DefenderActionEvent> defenderActionLog;
    private List<AlertEvent> alertLog;
    /** De-duplication key set "defenderId:nodeId" for first-observation alerts. */
    private Set<String> alertedPairs;

    // Decoupled timing driver (SimulationClock)
    private final SimulationClock clock;

    // Track current speed (ms per tick)
    private long currentPeriodMs = 50;

    // Track if topology was externally configured (should be preserved on reset)
    private boolean externalTopology = false;

    // --- Constructors ---
    public NetworkSimulation() {
        this(new NetworkTopology());
        // For default construction, generate a random topology as before
        initializeNetwork();
        initializeDefaultAgents(); // <-- ensure visible agents initially
    }

    public NetworkSimulation(NetworkTopology topology) {
        this.topology = topology;
        this.agents = new ArrayList<>();
        this.activeAttacks = new HashMap<>();
        this.eventLog = new ArrayList<>();
        this.alerts = new HashMap<>();
        this.trafficFlows = new HashMap<>();
        this.timeStep = 0;
        this.isRunning = false;

        // Instrumentation logs
        this.defenderActionLog = new ArrayList<>();
        this.alertLog = new ArrayList<>();
        this.alertedPairs = new HashSet<>();

        // Initialize decoupled clock and drive the simulation via ticks
        this.clock = new SimulationClock();
        this.clock.addListener(tick -> {
            if (isRunning) {
                step();
            }
        });
    }

    private void initializeNetwork() {
        topology.generateRandomTopology(20); // 20 nodes
    }

    /**
     * Add a couple of agents and assign their starting nodes so they render immediately.
     */
    private void initializeDefaultAgents() {
        if (topology == null || topology.getNodes().isEmpty()) return;

        // simple demo pair
        DefenderAgent defender = new DefenderAgent("defender-1");
        AttackerAgent attacker = new AttackerAgent("attacker-1");

        addAgent(defender);
        addAgent(attacker);

        // one no-op tick for each to set currentNodeId (their first act() initializes node)
        defender.act();
        attacker.act();
    }

    // --- Simulation control (driven by SimulationClock) ---

    /**
     * Starts (or restarts) the simulation at the last configured speed.
     */
    public void start() {
        start(currentPeriodMs);
    }

    /**
     * Starts (or restarts) the simulation at a specific period in milliseconds.
     */
    public void start(long periodMs) {
        currentPeriodMs = periodMs;
        if (!isRunning) {
            isRunning = true;
        }
        // Ensure something to show
        if (agents.isEmpty()) {
            initializeDefaultAgents();
        }
        // Restart the existing clock with the new period
        clock.stop();
        clock.start(periodMs);
    }

    public void stop() {
        isRunning = false;
        clock.stop();
    }

    public void pause() {
        isRunning = false;
        clock.pause();
    }

    /**
     * Resume with the last configured period.
     */
    public void resume() {
        if (!isRunning) {
            isRunning = true;
            clock.start(currentPeriodMs);
        }
    }

    /**
     * Change the tick period while running; persists as the new default speed.
     */
    public void setSpeed(long periodMs) {
        currentPeriodMs = periodMs;
        if (isRunning) {
            clock.stop();
            clock.start(periodMs);
        }
    }

    /**
     * Perform one simulation step (normally invoked by the clock).
     */
    public void step() {
        if (!isRunning) return;
        timeStep++;

        // Update agents
        for (SimulationAgent agent : agents) {
            agent.act();
        }

        // Update active attacks
        for (AttackScenario attack : activeAttacks.values()) {
            attack.step();
        }

        // Update network state
        updateNetworkState();

        // Clear old traffic flows
        cleanupTrafficFlows();
    }

    // Attack activation methods
    public void activatePortScanAttack() {
        activateAttack(new PortScanAttack(this, 0.7, 60));
    }

    public void activateDDoSAttack() {
        activateAttack(new DDoSAttack(this, 0.8, 120));
    }

    public void activateDataExfiltrationAttack() {
        activateAttack(new DataExfiltrationAttack(this, 0.6, 180));
    }

    public void activateLateralMovementAttack() {
        activateAttack(new LateralMovementAttack(this, 0.7, 150));
    }

    public void activateRansomwareAttack() {
        activateAttack(new RansomwareAttack(this, 0.8, 300));
    }

    public void activatePhishingAttack() {
        activateAttack(new PhishingAttack(this, 0.5, 240));
    }

    private void activateAttack(AttackScenario attack) {
        String attackName = attack.getName();
        // Deactivate existing attack of same type if any
        if (activeAttacks.containsKey(attackName)) {
            activeAttacks.get(attackName).deactivate();
        }
        activeAttacks.put(attackName, attack);
        attack.activate();
        addLogEntry("ATTACK_START", attackName + " initiated");
        triggerAlert("SECURITY_CENTER", attackName + " attack detected!");
    }

    public void deactivateAttack(String attackName) {
        AttackScenario attack = activeAttacks.get(attackName);
        if (attack != null) {
            attack.deactivate();
            activeAttacks.remove(attackName);
            addLogEntry("ATTACK_END", attackName + " terminated");
        }
    }

    public void addAgent(SimulationAgent agent) {
        agent.setEnvironment(this);
        agents.add(agent);
    }

    private void updateNetworkState() {
        // Update node states based on attacks and defenses
        for (NetworkTopology.NetworkNode node : topology.getNodes()) {
            // Natural recovery for defended nodes
            if (node.isDefended() && node.isCompromised()) {
                if (Math.random() < 0.1) {
                    node.setCompromised(false);
                    addLogEntry("RECOVERY", node.getId() + " recovered");
                }
            }
            // Clear temporary states
            if (node.isOverloaded() && !isUnderDDoS(node)) {
                node.setOverloaded(false);
                node.setAvailable(true);
            }
        }
    }

    private boolean isUnderDDoS(NetworkTopology.NetworkNode node) {
        AttackScenario ddos = activeAttacks.get("DDoS");
        return ddos != null && ddos.isActive()
                && ddos.getAffectedNodes().contains(node.getId());
    }

    // Node operations for RL agents
    public void isolateNode(String nodeId) {
        NetworkTopology.NetworkNode node = topology.getNodeMap().get(nodeId);
        if (node != null) {
            node.setIsolated(true);
            addLogEntry("ISOLATION", nodeId + " isolated from network");
        }
    }

    public void restoreNode(String nodeId) {
        NetworkTopology.NetworkNode node = topology.getNodeMap().get(nodeId);
        if (node != null) {
            node.setIsolated(false);
            addLogEntry("RESTORATION", nodeId + " restored to network");
        }
    }

    public void defendNode(String nodeId) {
        NetworkTopology.NetworkNode node = topology.getNodeMap().get(nodeId);
        if (node != null) {
            node.setDefended(true);
            node.setCompromised(false);
            addLogEntry("DEFENSE", nodeId + " defended");
        }
    }

    // Visualization helpers
    public void addTrafficFlow(String source, String target, double volume) {
        String key = source + "->" + target;
        trafficFlows.put(key, new TrafficFlow(source, target, volume, timeStep));
    }

    public void addInfectionPath(String source, String target) {
        addTrafficFlow(source, target, 100);
        NetworkTopology.NetworkNode targetNode = topology.getNodeMap().get(target);
        if (targetNode != null) {
            targetNode.setColor(Color.ORANGE);
        }
    }

    public void addMovementPath(String source, String target) {
        addTrafficFlow(source, target, 50);
    }

    private void cleanupTrafficFlows() {
        // Remove flows older than 10 time steps
        trafficFlows.entrySet().removeIf(entry ->
                timeStep - entry.getValue().timestamp > 10);
    }

    // Alert management
    public void triggerAlert(String source, String message) {
        Alert alert = new Alert(source, message, timeStep);
        alerts.computeIfAbsent(source, k -> new ArrayList<>()).add(alert);
        System.out.println("[ALERT] " + source + ": " + message);
    }

    public void displayAlert(String title, String message) {
        // UI layer may override; keep console logging here
        System.out.println("=== " + title + " ===\n" + message);
    }

    // Logging
    public void addLogEntry(String type, String message) {
        eventLog.add(new SimulationEvent(type, message, timeStep));
    }

    // --- Lifecycle helpers ---

    /**
     * Reset the simulation to initial state.
     *
     * Resets node states while preserving the configured topology structure.
     */
    public void reset() {
        // Ensure timer stops and tick is reset
        stop();
        clock.reset();

        // Reset node states without regenerating topology
        // This preserves the configured network size (servers, workstations, routers)
        if (topology != null) {
            topology.reset();  // Uses NetworkTopology.reset() which only resets node states
        }

        // Clear simulation state
        if (agents != null) agents.clear();
        if (activeAttacks != null) activeAttacks.clear();
        if (eventLog != null) eventLog.clear();
        if (alerts != null) alerts.clear();
        if (trafficFlows != null) trafficFlows.clear();
        clearInstrumentationLogs();
        timeStep = 0;

        // Note: We don't call initializeDefaultAgents() for RL experiments
        // The RL environment manages its own agent interactions
    }

    /**
     * Full reset that also regenerates the topology.
     * Use this only when you want to completely restart with a new network.
     */
    public void fullReset() {
        stop();
        clock.reset();

        // Regenerate topology (for GUI/demo use)
        topology = new NetworkTopology();
        initializeNetwork();

        if (agents != null) agents.clear();
        if (activeAttacks != null) activeAttacks.clear();
        if (eventLog != null) eventLog.clear();
        if (alerts != null) alerts.clear();
        if (trafficFlows != null) trafficFlows.clear();
        clearInstrumentationLogs();
        timeStep = 0;

        // Re-seed agents for visualization
        initializeDefaultAgents();
    }

    // --- Getters ---
    public NetworkTopology getTopology() {
        return topology;
    }

    public List<SimulationAgent> getAgents() {
        return agents;
    }

    public Map<String, AttackScenario> getActiveAttacks() {
        return activeAttacks;
    }

    public int getTimeStep() {
        return timeStep;
    }

    public Map<String, TrafficFlow> getTrafficFlows() {
        return trafficFlows;
    }

    public List<SimulationEvent> getEventLog() {
        return eventLog;
    }

    public boolean isRunning() {
        return isRunning;
    }

    public long getCurrentTick() {
        return clock.getCurrentTick();
    }

    public long getCurrentPeriodMs() {
        return currentPeriodMs;
    }

    // ================================================================
    //  Instrumentation
    // ================================================================
    //
    // Two append-only logs backing case-study metrics M8 and M9.
    //   * defenderActionLog — every observable defender action with a
    //     timestamp (simulation step). Used for M8 — total defender
    //     actions in t ∈ [65, 150].
    //   * alertLog — first-observation events, one per (defender, node)
    //     pair. Used for M9 — earliest detection tick in the interval.
    //
    // Semantics:
    //   * Both lists accumulate across step() calls.
    //   * logDefenderAction appends unconditionally.
    //   * logAlert deduplicates by "defenderId:observedNodeId" — repeat
    //     observations of the same (defender, node) pair are dropped.
    //   * Both lists are cleared on reset(), fullReset() and
    //     restoreFromSnapshot() so a resumed branch starts fresh.
    //   * Getters return defensive copies — callers cannot mutate
    //     internal state through the returned list.

    /**
     * Append a defender-action event to the instrumentation log.
     * Called by {@code DefenderAgent.emitAction}. No-op on null input.
     */
    public void logDefenderAction(DefenderActionEvent event) {
        if (event == null) return;
        if (defenderActionLog == null) defenderActionLog = new ArrayList<>();
        defenderActionLog.add(event);
    }

    /**
     * Append a first-observation alert, deduplicated by
     * {@code (defenderId, observedNodeId)}. Repeat observations of the
     * same pair are silently dropped — this is what makes the log usable
     * for M9 (one alert per defender/node pair).
     */
    public void logAlert(AlertEvent event) {
        if (event == null) return;
        if (alertLog == null) alertLog = new ArrayList<>();
        if (alertedPairs == null) alertedPairs = new HashSet<>();
        String key = (event.defenderId == null ? "?" : event.defenderId)
                + ":" + (event.observedNodeId == null ? "?" : event.observedNodeId);
        if (alertedPairs.add(key)) {
            alertLog.add(event);
        }
    }

    /**
     * Defensive copy of the defender-action log.
     * The returned list is a snapshot — safe to iterate while the
     * simulation continues running.
     */
    public List<DefenderActionEvent> getDefenderActionLog() {
        return (defenderActionLog == null)
                ? new ArrayList<>()
                : new ArrayList<>(defenderActionLog);
    }

    /**
     * Defensive copy of the first-observation alert log.
     */
    public List<AlertEvent> getAlertLog() {
        return (alertLog == null)
                ? new ArrayList<>()
                : new ArrayList<>(alertLog);
    }

    /**
     * Clear both instrumentation logs and the alert-pair dedup set.
     * Package-private — used internally by reset(), fullReset() and
     * restoreFromSnapshot(). Tests may invoke via reflection.
     */
    private void clearInstrumentationLogs() {
        if (defenderActionLog != null) defenderActionLog.clear();
        if (alertLog != null)          alertLog.clear();
        if (alertedPairs != null)      alertedPairs.clear();
    }

    // ================================================================
    //  Snapshot / restore
    // ================================================================
    //
    // These two methods enable mid-trajectory state capture and resume,
    // which the case-study runner (Task D) needs to branch from a single
    // baseline run at t=65 and continue under different defender configs.
    //
    // Fidelity notes:
    //   * Node states are restored exactly (compromised, defended,
    //     isolated, monitored, vulnerabilityScore, criticality, position,
    //     tags, available, overloaded, encrypted, performanceImpact).
    //   * Active attacks are reconstructed from class name + intensity +
    //     duration + elapsedTime via reflection. Attack-internal RNG
    //     state (e.g. PortScanAttack.scannedNodes) is NOT preserved —
    //     each reconstructed attack starts with a fresh Random.
    //   * Agents are reconstructed from class name + currentNodeId +
    //     role + health. Agent-internal state beyond position (e.g.
    //     confirmation counter for ThoroughDefenderAgent) is NOT
    //     preserved. The case-study runner REPLACES defenders anyway,
    //     so this is an acceptable loss.
    //   * Math.random() is process-global and unseeded, so any attack /
    //     agent that uses it will drift. See
    //     NetworkSimulationRestoreTest for empirical divergence data.
    //   * The topology STRUCTURE (edges) is not restored — it is
    //     assumed to be identical in the live instance.

    /**
     * Capture the current simulation state into a {@link NetworkStateSnapshot}.
     * Call on a paused (or freshly stopped) simulation for a consistent view.
     */
    public NetworkStateSnapshot snapshot() {
        NetworkStateSnapshot snap = new NetworkStateSnapshot();
        snap.timestamp = (clock != null) ? clock.getCurrentTick() : 0L;
        snap.timeStep = this.timeStep;

        if (topology != null) {
            snap.compromisedRatio = topology.getCompromisedRatio();
            snap.defendedRatio = topology.getDefendedRatio();
            for (NetworkTopology.NetworkNode node : topology.getNodes()) {
                snap.nodes.put(node.getId(), captureNodeSnapshot(node));
            }
            for (NetworkTopology.NetworkEdge edge : topology.getEdges()) {
                snap.edges.add(edge.getSource().getId()
                        + "->" + edge.getTarget().getId());
            }
        }

        if (activeAttacks != null) {
            for (AttackScenario attack : activeAttacks.values()) {
                snap.activeAttacks.add(captureAttackSnapshot(attack));
            }
        }
        if (agents != null) {
            for (SimulationAgent agent : agents) {
                snap.agents.add(captureAgentSnapshot(agent));
            }
        }
        return snap;
    }

    /**
     * Restore the simulation to a previously captured state.
     *
     * <p>Replaces all node states, active attacks, agents, and the clock
     * / timeStep. Clears logs, alerts and traffic flows (old events are
     * not meaningful for the resumed branch).
     *
     * @param snapshot the captured state to restore
     * @throws IllegalArgumentException if snapshot or its node map is null
     */
    public void restoreFromSnapshot(NetworkStateSnapshot snapshot) {
        if (snapshot == null) {
            throw new IllegalArgumentException("Snapshot cannot be null");
        }
        if (snapshot.nodes == null) {
            throw new IllegalArgumentException("Snapshot has no node data");
        }

        // 1) Stop the clock (idempotent — safe even if never started).
        stop();

        // 2) Restore node states onto the live topology.
        if (topology != null) {
            for (Map.Entry<String, NodeSnapshot> e : snapshot.nodes.entrySet()) {
                NetworkTopology.NetworkNode liveNode =
                        topology.getNodeMap().get(e.getKey());
                if (liveNode == null) continue;      // node missing in live topology
                applyNodeSnapshot(liveNode, e.getValue());
            }
        }

        // 3) Restore active attacks.
        if (activeAttacks == null) {
            activeAttacks = new HashMap<>();
        } else {
            // Deactivate any currently-running attacks before swapping.
            for (AttackScenario a : new ArrayList<>(activeAttacks.values())) {
                try { a.deactivate(); } catch (Exception ignored) { }
            }
            activeAttacks.clear();
        }
        if (snapshot.activeAttacks != null) {
            for (AttackSnapshot as : snapshot.activeAttacks) {
                AttackScenario rebuilt = reconstructAttackFromSnapshot(as);
                if (rebuilt != null) {
                    activeAttacks.put(rebuilt.getName(), rebuilt);
                }
            }
        }

        // 4) Restore agents.
        if (agents == null) {
            agents = new ArrayList<>();
        } else {
            agents.clear();
        }
        if (snapshot.agents != null) {
            for (AgentStateSnapshot gs : snapshot.agents) {
                SimulationAgent rebuilt = reconstructAgentFromSnapshot(gs);
                if (rebuilt != null) {
                    agents.add(rebuilt);
                }
            }
        }

        // 5) Restore clock + timeStep (but do NOT auto-start the scheduler).
        this.timeStep = snapshot.timeStep;
        if (clock != null) {
            clock.setCurrentTick(snapshot.timestamp);
        }

        // 6) Clear transient logs — the resumed run starts a fresh history.
        if (eventLog != null)     eventLog.clear();
        if (alerts != null)       alerts.clear();
        if (trafficFlows != null) trafficFlows.clear();
        clearInstrumentationLogs();
    }

    // -------- capture helpers --------

    private NodeSnapshot captureNodeSnapshot(NetworkTopology.NetworkNode node) {
        NodeSnapshot ns = new NodeSnapshot();
        ns.id = node.getId();
        ns.type = (node.getType() != null)
                ? node.getType().name() : NetworkTopology.NodeType.WORKSTATION.name();
        ns.compromised = node.isCompromised();
        ns.defended = node.isDefended();
        ns.isolated = node.isIsolated();
        ns.monitored = node.isMonitored();
        ns.vulnerabilityScore = node.getVulnerabilityScore();
        ns.criticality = node.getCriticality();
        if (node.getPosition() != null) {
            ns.x = node.getPosition().x;
            ns.y = node.getPosition().y;
        }
        // Tags: NetworkNode currently has no public getter; use reflection
        // to avoid modifying NetworkNode. Fall back to empty set on failure.
        ns.tags = new HashSet<>();
        Object tagsObj = reflectGetField(node, "tags");
        if (tagsObj instanceof java.util.Collection<?>) {
            for (Object t : (java.util.Collection<?>) tagsObj) {
                if (t instanceof String) ns.tags.add((String) t);
            }
        }
        return ns;
    }

    private AttackSnapshot captureAttackSnapshot(AttackScenario attack) {
        AttackSnapshot as = new AttackSnapshot();
        as.name = attack.getName();
        as.isActive = attack.isActive();
        as.affectedNodes = new ArrayList<>(attack.getAffectedNodes());
        as.metrics = new HashMap<>(attack.getMetrics());
        // Reconstruction metadata (underscore-prefixed keys avoid collisions
        // with the attack's own metric keys like "Total Nodes Scanned"):
        as.metrics.put("_className", attack.getClass().getName());
        as.metrics.put("_intensity",
                reflectGetField(attack, "intensity"));
        as.metrics.put("_duration",
                reflectGetField(attack, "duration"));
        as.metrics.put("_elapsedTime",
                reflectGetField(attack, "elapsedTime"));
        return as;
    }

    private AgentStateSnapshot captureAgentSnapshot(SimulationAgent agent) {
        AgentStateSnapshot gs = new AgentStateSnapshot();
        gs.agentId = agent.getId();
        gs.role = agent.getRole();
        gs.isBusy = false;
        gs.actionsPerformed = 0;
        gs.performance = agent.getHealth();
        gs.localKnowledge = new HashMap<>();
        gs.localKnowledge.put("_className", agent.getClass().getName());
        gs.localKnowledge.put("_health", agent.getHealth());
        gs.localKnowledge.put("_agentType",
                (agent.getType() != null) ? agent.getType().name() : "NEUTRAL");

        // Position — DefenderAgent and AttackerAgent both expose it.
        if (agent instanceof DefenderAgent) {
            gs.localKnowledge.put("_currentNodeId",
                    ((DefenderAgent) agent).getCurrentNodeId());
            Object str = reflectGetField(agent, "defenseStrength");
            if (str != null) gs.localKnowledge.put("_strength", str);
        } else if (agent instanceof AttackerAgent) {
            gs.localKnowledge.put("_currentNodeId",
                    ((AttackerAgent) agent).getCurrentNodeId());
            Object str = reflectGetField(agent, "attackStrength");
            if (str != null) gs.localKnowledge.put("_strength", str);
        }
        return gs;
    }

    // -------- apply helpers --------

    private void applyNodeSnapshot(NetworkTopology.NetworkNode liveNode,
                                   NodeSnapshot snap) {
        liveNode.setCompromised(snap.compromised);
        liveNode.setDefended(snap.defended);
        liveNode.setIsolated(snap.isolated);
        liveNode.setMonitored(snap.monitored);
        liveNode.setVulnerabilityScore(snap.vulnerabilityScore);
        liveNode.setCriticality(snap.criticality);
        if (liveNode.getPosition() != null) {
            liveNode.getPosition().setLocation(snap.x, snap.y);
        }
        // Reset per-snapshot transient flags that NodeSnapshot doesn't track.
        // The simulation step() re-evaluates overloaded/available each tick,
        // so we clear them here to avoid stale state.
        liveNode.setOverloaded(false);
        liveNode.setAvailable(true);
        liveNode.setEncrypted(false);
        liveNode.setPerformanceImpact(0.0);
        // Restore tags best-effort via reflection (no public setter).
        Object tagsObj = reflectGetField(liveNode, "tags");
        if (tagsObj instanceof java.util.Set<?>) {
            @SuppressWarnings("unchecked")
            java.util.Set<String> liveTags = (java.util.Set<String>) tagsObj;
            liveTags.clear();
            if (snap.tags != null) liveTags.addAll(snap.tags);
        }
    }

    @SuppressWarnings("unchecked")
    private AttackScenario reconstructAttackFromSnapshot(AttackSnapshot as) {
        if (as == null || as.metrics == null) return null;
        try {
            String className = (String) as.metrics.get("_className");
            if (className == null) return null;

            double intensity = toDouble(as.metrics.get("_intensity"), 0.5);
            int duration     = toInt(as.metrics.get("_duration"), 60);
            int elapsedTime  = toInt(as.metrics.get("_elapsedTime"), 0);

            Class<?> cls = Class.forName(className);
            Constructor<?> ctor = cls.getConstructor(
                    NetworkSimulation.class, double.class, int.class);
            AttackScenario rebuilt =
                    (AttackScenario) ctor.newInstance(this, intensity, duration);

            // Activate so subclass-specific internals (scannedNodes sets,
            // discoveredServices maps, etc.) are initialised, then re-apply
            // captured progress on top.
            rebuilt.activate();
            reflectSetField(rebuilt, "elapsedTime", elapsedTime);
            reflectSetField(rebuilt, "isActive", as.isActive);
            if (as.affectedNodes != null) {
                reflectSetField(rebuilt, "affectedNodes",
                        new ArrayList<>(as.affectedNodes));
            }
            return rebuilt;
        } catch (Exception e) {
            System.err.println("restoreFromSnapshot: failed to rebuild attack '"
                    + as.name + "': " + e);
            return null;
        }
    }

    private SimulationAgent reconstructAgentFromSnapshot(AgentStateSnapshot gs) {
        if (gs == null || gs.localKnowledge == null) return null;
        try {
            String className = (String) gs.localKnowledge.get("_className");
            if (className == null) return null;

            Class<?> cls = Class.forName(className);

            SimulationAgent agent;
            Double strength = (gs.localKnowledge.get("_strength") instanceof Number)
                    ? ((Number) gs.localKnowledge.get("_strength")).doubleValue()
                    : null;
            try {
                if (strength != null) {
                    Constructor<?> ctor =
                            cls.getConstructor(String.class, double.class);
                    agent = (SimulationAgent)
                            ctor.newInstance(gs.agentId, strength);
                } else {
                    throw new NoSuchMethodException();
                }
            } catch (NoSuchMethodException ex) {
                Constructor<?> ctor = cls.getConstructor(String.class);
                agent = (SimulationAgent) ctor.newInstance(gs.agentId);
            }

            agent.setEnvironment(this);
            if (gs.role != null) agent.setRole(gs.role);
            Object h = gs.localKnowledge.get("_health");
            if (h instanceof Number) agent.setHealth(((Number) h).doubleValue());

            String nodeId = (String) gs.localKnowledge.get("_currentNodeId");
            if (nodeId != null) {
                // Works for both DefenderAgent and AttackerAgent (same field name).
                reflectSetField(agent, "currentNodeId", nodeId);
            }
            return agent;
        } catch (Exception e) {
            System.err.println("restoreFromSnapshot: failed to rebuild agent '"
                    + gs.agentId + "': " + e);
            return null;
        }
    }

    // -------- tiny reflection utilities --------

    private static Object reflectGetField(Object target, String fieldName) {
        if (target == null) return null;
        try {
            Field f = findDeclaredField(target.getClass(), fieldName);
            f.setAccessible(true);
            return f.get(target);
        } catch (Exception e) {
            return null;
        }
    }

    private static void reflectSetField(Object target, String fieldName, Object value) {
        if (target == null) return;
        try {
            Field f = findDeclaredField(target.getClass(), fieldName);
            f.setAccessible(true);
            f.set(target, value);
        } catch (Exception e) {
            System.err.println("reflectSetField(" + fieldName + ") failed: " + e);
        }
    }

    private static Field findDeclaredField(Class<?> cls, String name)
            throws NoSuchFieldException {
        Class<?> c = cls;
        while (c != null && c != Object.class) {
            try {
                return c.getDeclaredField(name);
            } catch (NoSuchFieldException ex) {
                c = c.getSuperclass();
            }
        }
        throw new NoSuchFieldException(name + " not found on " + cls.getName());
    }

    private static double toDouble(Object o, double fallback) {
        if (o instanceof Number) return ((Number) o).doubleValue();
        return fallback;
    }

    private static int toInt(Object o, int fallback) {
        if (o instanceof Number) return ((Number) o).intValue();
        return fallback;
    }

    // Helper classes
    public static class TrafficFlow {
        public String source;
        public String target;
        public double volume;
        public int timestamp;

        public TrafficFlow(String source, String target, double volume, int timestamp) {
            this.source = source;
            this.target = target;
            this.volume = volume;
            this.timestamp = timestamp;
        }
    }

    public static class Alert {
        public String source;
        public String message;
        public int timestamp;

        public Alert(String source, String message, int timestamp) {
            this.source = source;
            this.message = message;
            this.timestamp = timestamp;
        }
    }

    public static class SimulationEvent {
        public String type;
        public String message;
        public int timestamp;

        public SimulationEvent(String type, String message, int timestamp) {
            this.type = type;
            this.message = message;
            this.timestamp = timestamp;
        }
    }
}