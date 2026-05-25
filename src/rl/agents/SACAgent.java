package rl.agents;

import rl.ReinforcementLearningFramework.*;
import rl.persistence.AgentDTO;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Soft Actor-Critic (SAC) agent for discrete action spaces.
 *
 * SAC-Discrete adaptation (Christodoulou 2019) of the continuous
 * SAC framework (Haarnoja et al. 2018). Components:
 *  - Twin Q-networks with the clipped double-Q operator
 *  - Softmax policy network (actor)
 *  - Target Q-networks with Polyak averaging
 *  - Automatic entropy temperature (alpha) tuning
 *  - Off-policy experience replay buffer
 *  - Reward and input normalization via Welford's online algorithm
 */
public class SACAgent implements Agent {

    // ==================== Core Parameters ====================
    private final int stateSize;
    private final int actionSize;
    private double actorLearningRate;
    private double criticLearningRate;
    private final double discountFactor;     // γ (default 0.99)

    // ==================== SAC-Specific Parameters ====================
    private double logAlpha;                 // log of temperature parameter
    private double alpha;                    // Temperature (entropy weight)
    private double alphaLearningRate;        // Learning rate for α optimization
    private final double targetEntropy;      // Target entropy = -log(1/|A|) * ratio
    private final double softUpdateTau;      // τ for Polyak averaging (default 0.005)

    // ==================== Networks ====================
    private final SACNetwork actorNetwork;   // Policy π(a|s)
    private final SACNetwork critic1Network; // Q₁(s, a)
    private final SACNetwork critic2Network; // Q₂(s, a)
    private final SACNetwork target1Network; // Q̄₁(s, a) — frozen target
    private final SACNetwork target2Network; // Q̄₂(s, a) — frozen target

    // ==================== Experience Replay ====================
    private final Experience[] replayBuffer;  // Fixed-size circular array
    private final int bufferCapacity;
    private int bufferIndex = 0;              // Next write position
    private int bufferSize = 0;               // Current number of stored experiences
    private final int batchSize;
    private final int warmupSteps;           // Don't train until buffer has this many samples

    // ==================== Input Normalization (Welford's) ====================
    private final double[] inputMean;
    private final double[] inputM2;
    private long inputCount = 0;

    // ==================== Reward Normalization (Welford's) ====================
    private double rewardMean = 0.0;
    private double rewardM2 = 0.0;
    private long rewardCount = 0;
    private static final double REWARD_CLIP = 10.0;

    // ==================== Statistics ====================
    private int episodeCount = 0;
    private int totalSteps = 0;
    private double recentActorLoss = 0.0;
    private double recentCriticLoss = 0.0;
    private double recentAlphaLoss = 0.0;
    private double recentEntropy = 0.0;
    private final Random random;

    // ==================== Constructors ====================

    /**
     * Full constructor with all SAC hyperparameters.
     */
    public SACAgent(int stateSize, int actionSize, double actorLR, double criticLR,
                    double discountFactor, double initialAlpha, double alphaLR,
                    double targetEntropyRatio, double tau, int bufferCapacity,
                    int batchSize, int warmupSteps) {
        this.stateSize = stateSize;
        this.actionSize = actionSize;
        this.actorLearningRate = actorLR;
        this.criticLearningRate = criticLR;
        this.discountFactor = discountFactor;
        this.softUpdateTau = tau;
        this.bufferCapacity = bufferCapacity;
        this.batchSize = batchSize;
        this.warmupSteps = warmupSteps;
        this.random = new Random();

        // Entropy temperature (auto-tuned)
        this.alpha = initialAlpha;
        this.logAlpha = Math.log(Math.max(alpha, 1e-10));
        this.alphaLearningRate = alphaLR;
        // Target entropy: -log(1/|A|) * ratio = log(|A|) * ratio
        this.targetEntropy = -Math.log(1.0 / actionSize) * targetEntropyRatio;

        // Adaptive hidden sizes
        int hidden1 = Math.min(256, Math.max(64, stateSize * 2));
        int hidden2 = Math.min(128, Math.max(32, hidden1 / 2));

        // Actor: outputs action logits (softmax applied externally)
        int[] actorLayers = {stateSize, hidden1, hidden2, actionSize};
        this.actorNetwork = new SACNetwork(actorLayers, actorLR);

        // Twin critics: each outputs Q(s, a) for all actions
        int[] criticLayers = {stateSize, hidden1, hidden2, actionSize};
        this.critic1Network = new SACNetwork(criticLayers, criticLR);
        this.critic2Network = new SACNetwork(criticLayers, criticLR);

        // Target networks (initialized as copies)
        this.target1Network = new SACNetwork(criticLayers, criticLR);
        this.target2Network = new SACNetwork(criticLayers, criticLR);
        target1Network.copyWeightsFrom(critic1Network);
        target2Network.copyWeightsFrom(critic2Network);

        // Circular replay buffer (O(1) insertion)
        this.replayBuffer = new Experience[bufferCapacity];

        // Input normalization
        this.inputMean = new double[stateSize];
        this.inputM2 = new double[stateSize];

        System.out.println("[SACAgent] Initialized v1.1 with actor=" + Arrays.toString(actorLayers)
                + ", critic=" + Arrays.toString(criticLayers)
                + ", α=" + String.format("%.4f", alpha)
                + ", targetH=" + String.format("%.3f", targetEntropy)
                + ", τ=" + tau + ", buffer=" + bufferCapacity);
    }

    /**
     * Standard constructor with sensible defaults for cybersecurity environments.
     */
    public SACAgent(int stateSize, int actionSize, double learningRate, double discountFactor) {
        this(stateSize, actionSize,
                learningRate * 0.5,   // Actor LR (slower than critic)
                learningRate,          // Critic LR
                discountFactor,
                0.2,                   // Initial α (entropy temperature)
                learningRate * 0.5,    // Alpha LR
                0.5,                   // Target entropy ratio (50% of max entropy)
                0.005,                 // Polyak τ
                20000,                 // Buffer capacity
                64,                    // Batch size
                256                    // Warmup steps
        );
    }

    // ==================== Agent Interface ====================

    @Override
    public Action selectAction(State state) {
        totalSteps++;

        // Update input normalization
        updateInputStats(state.getFeatures());
        double[] normalizedState = normalizeInput(state.getFeatures());

        // Get action probabilities from actor
        double[] logits = actorNetwork.forward(normalizedState);
        double[] probs = softmax(logits);

        // Sample action from policy distribution
        int actionIndex = sampleFromDistribution(probs);

        return new Action(actionIndex, "sac_action");
    }

    @Override
    public void updatePolicy(EpisodeResult episode) {
        episodeCount++;

        // Add all transitions to replay buffer
        List<Transition> transitions = episode.getTransitions();
        for (int i = 0; i < transitions.size(); i++) {
            Transition t = transitions.get(i);
            boolean done = (i == transitions.size() - 1);

            updateRewardStats(t.reward);
            double normalizedReward = normalizeReward(t.reward);

            addExperience(t.state, t.action, normalizedReward, t.nextState, done);
        }

        // Only train after warmup
        if (bufferSize < warmupSteps) return;

        // Train on multiple batches per episode (more sample-efficient than DQN)
        int trainIterations = Math.min(transitions.size(), 10);

        double totalActorLoss = 0, totalCriticLoss = 0, totalAlphaLoss = 0, totalEntropy = 0;

        for (int i = 0; i < trainIterations; i++) {
            double[] losses = trainOnBatch();
            totalActorLoss += losses[0];
            totalCriticLoss += losses[1];
            totalAlphaLoss += losses[2];
            totalEntropy += losses[3];
        }

        if (trainIterations > 0) {
            recentActorLoss = totalActorLoss / trainIterations;
            recentCriticLoss = totalCriticLoss / trainIterations;
            recentAlphaLoss = totalAlphaLoss / trainIterations;
            recentEntropy = totalEntropy / trainIterations;
        }
    }

    // ==================== SAC Training ====================

    /**
     * SAC training on one mini-batch from replay buffer.
     *
     * 1. Update critics: minimize TD error with clipped double-Q
     * 2. Update actor: maximize Q + α * H(π)
     * 3. Update α: adjust temperature to target entropy
     * 4. Soft update target networks
     *
     * @return [actorLoss, criticLoss, alphaLoss, entropy]
     */
    private double[] trainOnBatch() {
        if (bufferSize < batchSize) {
            return new double[]{0, 0, 0, 0};
        }

        // Sample mini-batch from circular buffer
        List<Experience> batch = new ArrayList<>();
        for (int i = 0; i < batchSize; i++) {
            int idx = random.nextInt(bufferSize);
            batch.add(replayBuffer[idx]);
        }

        double batchCriticLoss = 0;
        double batchActorLoss = 0;
        double batchAlphaLoss = 0;
        double batchEntropy = 0;

        for (Experience exp : batch) {
            double[] normState = normalizeInput(exp.state.getFeatures());

            // ==================== 1. Critic Update ====================
            // Compute target Q-value using target networks
            double targetQ;
            if (exp.done || exp.nextState == null) {
                targetQ = exp.reward;
            } else {
                double[] normNextState = normalizeInput(exp.nextState.getFeatures());

                // Get next action distribution from current policy
                double[] nextLogits = actorNetwork.forward(normNextState);
                double[] nextProbs = softmax(nextLogits);

                // Target Q = Σ_a' π(a'|s') * [min(Q̄₁(s',a'), Q̄₂(s',a')) - α * log π(a'|s')]
                double[] targetQ1 = target1Network.forward(normNextState);
                double[] targetQ2 = target2Network.forward(normNextState);

                double nextV = 0;
                for (int a = 0; a < actionSize; a++) {
                    double minQ = Math.min(targetQ1[a], targetQ2[a]);
                    double logProb = Math.log(Math.max(nextProbs[a], 1e-10));
                    nextV += nextProbs[a] * (minQ - alpha * logProb);
                }

                targetQ = exp.reward + discountFactor * nextV;
            }

            // Clip target for stability
            targetQ = Math.max(-100, Math.min(100, targetQ));

            // Update both critics toward target
            double[] q1Values = critic1Network.forward(normState);
            double[] q2Values = critic2Network.forward(normState);

            int actionId = exp.action.getId();
            if (actionId >= 0 && actionId < actionSize) {
                // Critic 1 target (Huber loss via target manipulation)
                double[] target1 = q1Values.clone();
                double error1 = targetQ - target1[actionId];
                if (Math.abs(error1) <= 1.0) {
                    target1[actionId] = targetQ;
                } else {
                    target1[actionId] += Math.signum(error1);
                }
                critic1Network.train(normState, target1);
                batchCriticLoss += Math.min(error1 * error1, Math.abs(error1));

                // Critic 2 target
                double[] target2 = q2Values.clone();
                double error2 = targetQ - target2[actionId];
                if (Math.abs(error2) <= 1.0) {
                    target2[actionId] = targetQ;
                } else {
                    target2[actionId] += Math.signum(error2);
                }
                critic2Network.train(normState, target2);
                batchCriticLoss += Math.min(error2 * error2, Math.abs(error2));
            }

            // ==================== 2. Actor Update ====================
            // Maximize: Σ_a π(a|s) * [min(Q₁(s,a), Q₂(s,a)) - α * log π(a|s)]
            double[] logits = actorNetwork.forward(normState);
            double[] probs = softmax(logits);

            q1Values = critic1Network.forward(normState);
            q2Values = critic2Network.forward(normState);

            // Compute policy entropy
            double entropy = 0;
            for (int a = 0; a < actionSize; a++) {
                if (probs[a] > 1e-10) {
                    entropy -= probs[a] * Math.log(probs[a]);
                }
            }
            batchEntropy += entropy;

            // Actor gradient via softmax Jacobian:
            // ∂L_π/∂z_a = Σ_b p_b * (δ_{ab} - p_a) * [α(log p_b + 1) - min(Q1_b, Q2_b)]
            double[] actorTargets = new double[actionSize];
            for (int a = 0; a < actionSize; a++) {
                double grad = 0;
                for (int b = 0; b < actionSize; b++) {
                    double indicator = (a == b) ? 1.0 : 0.0;
                    double jacobian = probs[b] * (indicator - probs[a]);
                    double minQb = Math.min(q1Values[b], q2Values[b]);
                    double logProbB = Math.log(Math.max(probs[b], 1e-10));
                    // Gradient of L_π = Σ p(a)[α log p(a) - Q(a)]
                    // We want to MINIMIZE this, so push logits in -∂L/∂z direction
                    // But since train() does output - target, setting target = logit + grad
                    // makes error = -grad, and update = lr * grad (correct descent on L_π)
                    double advB = minQb - alpha * (logProbB + 1.0);
                    grad += jacobian * advB;
                }

                // Actor learning rate is applied inside train(), not here
                actorTargets[a] = logits[a] + grad;
            }
            actorNetwork.train(normState, actorTargets);

            // Compute actor loss for monitoring
            double actorObjective = 0;
            for (int a = 0; a < actionSize; a++) {
                double minQ = Math.min(q1Values[a], q2Values[a]);
                double logProb = Math.log(Math.max(probs[a], 1e-10));
                actorObjective += probs[a] * (alpha * logProb - minQ);
            }
            batchActorLoss += actorObjective;

            // ==================== 3. Temperature (α) Update ====================
            // SAC dual gradient descent (Haarnoja et al. 2018):
            // J(α) = E[α * (H(π(·|s)) - H̄)]   — minimize
            // ∂J/∂log(α) = α * (H(π) - H̄)
            // If entropy > target → gradient positive → log_α decreases → α decreases ✓
            // If entropy < target → gradient negative → log_α increases → α increases ✓
            double alphaGradient = alpha * (entropy - targetEntropy);
            logAlpha -= alphaLearningRate * alphaGradient;
            logAlpha = Math.max(-5.0, Math.min(2.0, logAlpha));  // Clip log(α) ∈ [e⁻⁵, e²]
            alpha = Math.exp(logAlpha);
            batchAlphaLoss += alpha * (entropy - targetEntropy);
        }

        // ==================== 4. Soft Update Target Networks ====================
        target1Network.softCopyFrom(critic1Network, softUpdateTau);
        target2Network.softCopyFrom(critic2Network, softUpdateTau);

        int n = Math.max(1, batch.size());
        return new double[]{batchActorLoss / n, batchCriticLoss / n,
                batchAlphaLoss / n, batchEntropy / n};
    }

    // ==================== Experience Replay ====================

    /**
     * Add experience to circular replay buffer.
     * O(1) insertion — replaces oldest experience when full.
     */
    private void addExperience(State state, Action action, double reward,
                               State nextState, boolean done) {
        replayBuffer[bufferIndex % bufferCapacity] = new Experience(state, action, reward, nextState, done);
        bufferIndex++;
        if (bufferSize < bufferCapacity) {
            bufferSize++;
        }
    }

    // ==================== Input Normalization (Welford's) ====================

    private void updateInputStats(double[] features) {
        inputCount++;
        int n = Math.min(features.length, stateSize);
        for (int i = 0; i < n; i++) {
            double delta = features[i] - inputMean[i];
            inputMean[i] += delta / inputCount;
            double delta2 = features[i] - inputMean[i];
            inputM2[i] += delta * delta2;
        }
    }

    private double[] normalizeInput(double[] features) {
        double[] normalized = new double[stateSize];
        int n = Math.min(features.length, stateSize);

        if (inputCount < 2) {
            double maxAbs = 1.0;
            for (int i = 0; i < n; i++) {
                maxAbs = Math.max(maxAbs, Math.abs(features[i]));
            }
            for (int i = 0; i < n; i++) {
                normalized[i] = features[i] / maxAbs;
            }
            return normalized;
        }

        for (int i = 0; i < n; i++) {
            double variance = inputM2[i] / (inputCount - 1);
            double std = Math.sqrt(variance + 1e-8);
            normalized[i] = (features[i] - inputMean[i]) / std;
            normalized[i] = Math.max(-5.0, Math.min(5.0, normalized[i]));
        }
        return normalized;
    }

    // ==================== Reward Normalization (Welford's) ====================

    private void updateRewardStats(double reward) {
        rewardCount++;
        double delta = reward - rewardMean;
        rewardMean += delta / rewardCount;
        double delta2 = reward - rewardMean;
        rewardM2 += delta * delta2;
    }

    private double normalizeReward(double reward) {
        if (rewardCount < 2) return 0.0;
        double variance = rewardM2 / (rewardCount - 1);
        double std = Math.sqrt(variance + 1e-8);
        double normalized = (reward - rewardMean) / std;
        return Math.max(-REWARD_CLIP, Math.min(REWARD_CLIP, normalized));
    }

    // ==================== Softmax & Sampling ====================

    private double[] softmax(double[] logits) {
        double maxLogit = Double.NEGATIVE_INFINITY;
        for (double l : logits) maxLogit = Math.max(maxLogit, l);

        double[] exp = new double[logits.length];
        double sum = 0;
        for (int i = 0; i < logits.length; i++) {
            double clipped = Math.max(-20, Math.min(20, logits[i] - maxLogit));
            exp[i] = Math.exp(clipped);
            sum += exp[i];
        }
        if (sum < 1e-10) sum = 1e-10;
        for (int i = 0; i < exp.length; i++) exp[i] /= sum;
        return exp;
    }

    private int sampleFromDistribution(double[] probs) {
        double rand = random.nextDouble();
        double cumSum = 0;
        for (int i = 0; i < probs.length; i++) {
            cumSum += probs[i];
            if (rand < cumSum) return i;
        }
        return probs.length - 1;
    }

    // ==================== Persistence ====================

    @Override
    public void saveModel(String path) {
        System.out.println("[SACAgent] Saving model to: " + path);
    }

    @Override
    public void loadModel(String path) {
        System.out.println("[SACAgent] Loading model from: " + path);
    }

    public AgentDTO toDTO(String agentId, String environment) {
        Map<String, Object> config = new HashMap<>();
        config.put("stateSize", stateSize);
        config.put("actionSize", actionSize);
        config.put("actorLearningRate", actorLearningRate);
        config.put("criticLearningRate", criticLearningRate);
        config.put("discountFactor", discountFactor);
        config.put("alpha", alpha);
        config.put("logAlpha", logAlpha);
        config.put("targetEntropy", targetEntropy);
        config.put("softUpdateTau", softUpdateTau);
        config.put("episodeCount", episodeCount);
        config.put("totalSteps", totalSteps);
        config.put("rewardMean", rewardMean);
        config.put("rewardM2", rewardM2);
        config.put("rewardCount", rewardCount);

        return new AgentDTO.Builder("SAC", environment)
                .withAgentId(agentId)
                .withConfig(config)
                .withStatistics(episodeCount, recentActorLoss, recentCriticLoss)
                .build();
    }

    public static SACAgent fromDTO(AgentDTO dto) {
        Map<String, Object> config = dto.config;
        int stateSize = ((Number) config.get("stateSize")).intValue();
        int actionSize = ((Number) config.get("actionSize")).intValue();
        double actorLR = config.containsKey("actorLearningRate") ?
                ((Number) config.get("actorLearningRate")).doubleValue() : 0.0005;
        double criticLR = config.containsKey("criticLearningRate") ?
                ((Number) config.get("criticLearningRate")).doubleValue() : 0.001;
        double discount = ((Number) config.get("discountFactor")).doubleValue();

        double alphaVal = config.containsKey("alpha") ?
                ((Number) config.get("alpha")).doubleValue() : 0.2;
        double tau = config.containsKey("softUpdateTau") ?
                ((Number) config.get("softUpdateTau")).doubleValue() : 0.005;

        SACAgent agent = new SACAgent(stateSize, actionSize,
                actorLR,
                criticLR,
                discount,
                alphaVal,
                actorLR,     // alphaLR = actorLR (reasonable default)
                0.5,         // targetEntropyRatio
                tau,
                20000,       // buffer capacity
                64,          // batch size
                256          // warmup
        );

        if (config.containsKey("logAlpha")) {
            agent.logAlpha = ((Number) config.get("logAlpha")).doubleValue();
            agent.alpha = Math.exp(agent.logAlpha);
        }
        if (config.containsKey("rewardMean")) {
            agent.rewardMean = ((Number) config.get("rewardMean")).doubleValue();
        }
        if (config.containsKey("rewardM2")) {
            agent.rewardM2 = ((Number) config.get("rewardM2")).doubleValue();
        }
        if (config.containsKey("rewardCount")) {
            agent.rewardCount = ((Number) config.get("rewardCount")).longValue();
        }

        return agent;
    }

    // ==================== Getters ====================

    public int getStateSize() { return stateSize; }
    public int getActionSize() { return actionSize; }
    public double getActorLearningRate() { return actorLearningRate; }
    public double getCriticLearningRate() { return criticLearningRate; }
    public double getDiscountFactor() { return discountFactor; }
    public double getAlpha() { return alpha; }
    public double getTargetEntropy() { return targetEntropy; }
    public int getEpisodeCount() { return episodeCount; }
    public int getTotalSteps() { return totalSteps; }
    public double getRecentActorLoss() { return recentActorLoss; }
    public double getRecentCriticLoss() { return recentCriticLoss; }
    public double getRecentEntropy() { return recentEntropy; }
    public double getRewardMean() { return rewardMean; }
    public double getRewardStd() {
        return rewardCount > 1 ? Math.sqrt(rewardM2 / (rewardCount - 1)) : 0;
    }
    public int getReplayBufferSize() { return bufferSize; }

    // ==================== Experience Class ====================

    private static class Experience {
        final State state;
        final Action action;
        final double reward;
        final State nextState;
        final boolean done;

        Experience(State state, Action action, double reward, State nextState, boolean done) {
            this.state = state;
            this.action = action;
            this.reward = reward;
            this.nextState = nextState;
            this.done = done;
        }
    }

    // ==================== SAC Neural Network ====================

    /**
     * Neural network for SAC actor and critics.
     * Leaky ReLU hidden layers, linear output.
     * Supports soft (Polyak) updates for target networks.
     */
    static class SACNetwork {
        private final int[] layerSizes;
        double[][] weights;
        double[][] biases;
        private double learningRate;
        private final Random random;

        private double[][] activations;
        private double[][] preActivations;

        private static final double LEAKY_ALPHA = 0.01;
        private static final double GRADIENT_CLIP = 1.0;

        SACNetwork(int[] layers, double learningRate) {
            this.layerSizes = layers.clone();
            this.learningRate = learningRate;
            this.random = new Random(42);

            int numLayers = layers.length;
            weights = new double[numLayers - 1][];
            biases = new double[numLayers - 1][];

            for (int i = 0; i < numLayers - 1; i++) {
                int inputSize = layers[i];
                int outputSize = layers[i + 1];

                // He initialization
                double scale = Math.sqrt(2.0 / inputSize);
                weights[i] = new double[inputSize * outputSize];
                for (int j = 0; j < weights[i].length; j++) {
                    weights[i][j] = random.nextGaussian() * scale;
                }

                biases[i] = new double[outputSize];
                for (int j = 0; j < biases[i].length; j++) {
                    biases[i][j] = 0.01;
                }
            }

            activations = new double[numLayers][];
            preActivations = new double[numLayers][];
            for (int i = 0; i < numLayers; i++) {
                activations[i] = new double[layers[i]];
                preActivations[i] = new double[layers[i]];
            }
        }

        double[] forward(double[] input) {
            if (input == null || input.length == 0) {
                return new double[layerSizes[layerSizes.length - 1]];
            }

            int inputSize = Math.min(input.length, layerSizes[0]);
            Arrays.fill(activations[0], 0);
            System.arraycopy(input, 0, activations[0], 0, inputSize);
            System.arraycopy(activations[0], 0, preActivations[0], 0, activations[0].length);

            for (int layer = 0; layer < weights.length; layer++) {
                int inSize = layerSizes[layer];
                int outSize = layerSizes[layer + 1];

                for (int j = 0; j < outSize; j++) {
                    double sum = biases[layer][j];
                    for (int i = 0; i < inSize; i++) {
                        sum += activations[layer][i] * weights[layer][i * outSize + j];
                    }
                    preActivations[layer + 1][j] = sum;

                    if (layer < weights.length - 1) {
                        activations[layer + 1][j] = sum > 0 ? sum : LEAKY_ALPHA * sum;
                    } else {
                        activations[layer + 1][j] = sum;
                    }
                }
            }

            return activations[activations.length - 1].clone();
        }

        void train(double[] input, double[] targets) {
            forward(input);

            int numLayers = layerSizes.length;
            double[][] deltas = new double[numLayers][];
            for (int i = 0; i < numLayers; i++) {
                deltas[i] = new double[layerSizes[i]];
            }

            int outputLayer = numLayers - 1;
            for (int j = 0; j < layerSizes[outputLayer]; j++) {
                double error = activations[outputLayer][j] - targets[j];
                error = Math.max(-GRADIENT_CLIP * 10, Math.min(GRADIENT_CLIP * 10, error));
                deltas[outputLayer][j] = error;
            }

            for (int layer = numLayers - 2; layer > 0; layer--) {
                int currentSize = layerSizes[layer];
                int nextSize = layerSizes[layer + 1];

                for (int i = 0; i < currentSize; i++) {
                    double sum = 0;
                    for (int j = 0; j < nextSize; j++) {
                        sum += deltas[layer + 1][j] * weights[layer][i * nextSize + j];
                    }
                    double deriv = preActivations[layer][i] > 0 ? 1.0 : LEAKY_ALPHA;
                    deltas[layer][i] = sum * deriv;
                }
            }

            for (int layer = 0; layer < weights.length; layer++) {
                int inSize = layerSizes[layer];
                int outSize = layerSizes[layer + 1];

                for (int j = 0; j < outSize; j++) {
                    biases[layer][j] -= learningRate * deltas[layer + 1][j];

                    for (int i = 0; i < inSize; i++) {
                        double gradient = deltas[layer + 1][j] * activations[layer][i];
                        gradient = Math.max(-GRADIENT_CLIP, Math.min(GRADIENT_CLIP, gradient));
                        weights[layer][i * outSize + j] -= learningRate * gradient;
                    }
                }
            }
        }

        void copyWeightsFrom(SACNetwork other) {
            for (int i = 0; i < weights.length; i++) {
                System.arraycopy(other.weights[i], 0, weights[i], 0, weights[i].length);
                System.arraycopy(other.biases[i], 0, biases[i], 0, biases[i].length);
            }
        }

        void softCopyFrom(SACNetwork other, double tau) {
            for (int i = 0; i < weights.length; i++) {
                for (int j = 0; j < weights[i].length; j++) {
                    weights[i][j] = tau * other.weights[i][j] + (1.0 - tau) * weights[i][j];
                }
                for (int j = 0; j < biases[i].length; j++) {
                    biases[i][j] = tau * other.biases[i][j] + (1.0 - tau) * biases[i][j];
                }
            }
        }

        void setLearningRate(double lr) { this.learningRate = lr; }
        int[] getLayerSizes() { return layerSizes.clone(); }
    }
}