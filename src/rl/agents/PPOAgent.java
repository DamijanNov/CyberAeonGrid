package rl.agents;

import rl.ReinforcementLearningFramework.*;
import rl.persistence.AgentDTO;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;

/**
 * Proximal Policy Optimization (PPO) agent for discrete action spaces.
 *
 * Implements PPO-Clip as described in Schulman et al. 2017
 * ("Proximal Policy Optimization Algorithms", arXiv:1707.06347).
 *
 * Features:
 *  - Clipped surrogate objective with epsilon = 0.2
 *  - Generalized Advantage Estimation (GAE-lambda)
 *  - Entropy bonus for exploration
 *  - Reward and input normalization via Welford's online algorithm
 */
public class PPOAgent implements Agent {

    // ==================== Core Parameters ====================
    private final int stateSize;
    private final int actionSize;
    private double learningRate;
    private final double initialLearningRate;
    private final double discountFactor;   // gamma

    // ==================== PPO-Specific Parameters ====================
    private final double clipEpsilon;      // eps for clipping ratio (default 0.2)
    private final double gaeLambda;        // lambda for GAE (default 0.95)
    private final int ppoEpochs;           // K epochs per update (default 4)
    private final int miniBatchSize;       // Mini-batch size (default 64)
    private double entropyCoeff;           // c2 entropy bonus coefficient
    private final double valueLossCoeff;   // c1 value loss coefficient (default 0.5)
    private final double maxGradNorm;      // Gradient clipping norm (default 0.5)

    // ==================== Networks ====================
    private final ActorCriticNetwork actorNetwork;
    private final ActorCriticNetwork criticNetwork;

    // ==================== Trajectory Buffer ====================
    private final List<TrajectoryStep> trajectoryBuffer;

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
    private double recentPolicyLoss = 0.0;
    private double recentValueLoss = 0.0;
    private double recentEntropy = 0.0;
    private final Random random;

    // ==================== IT-Diag: Branch-frequency counters ====================
    // Each counter increments ONCE per trajectory step inside updateOnMiniBatch
    // (not per action). Together they let us see how often the unclipped-equality
    // case fired — the bug region in the publication implementation.
    private long diagSurrLess = 0;     // surr1 < surr2  (one of the two genuinely-unclipped cases)
    private long diagSurrEqual = 0;    // surr1 == surr2 (bug region: unclipped interior — was zeroed by old code)
    private long diagSurrGreater = 0;  // surr1 > surr2  (genuinely clipped — zero gradient is correct)

    // ==================== Constructors ====================

    /**
     * Full constructor with all PPO hyperparameters.
     */
    public PPOAgent(int stateSize, int actionSize, double learningRate,
                    double discountFactor, double clipEpsilon, double gaeLambda,
                    int ppoEpochs, int miniBatchSize, double entropyCoeff) {
        this.stateSize = stateSize;
        this.actionSize = actionSize;
        this.learningRate = learningRate;
        this.initialLearningRate = learningRate;
        this.discountFactor = discountFactor;
        this.clipEpsilon = clipEpsilon;
        this.gaeLambda = gaeLambda;
        this.ppoEpochs = ppoEpochs;
        this.miniBatchSize = miniBatchSize;
        this.entropyCoeff = entropyCoeff;
        this.valueLossCoeff = 0.5;
        this.maxGradNorm = 0.5;
        this.random = new Random();

        // Adaptive hidden sizes (identical to publication PPOAgent)
        int hidden1 = Math.min(128, Math.max(32, stateSize));
        int hidden2 = Math.min(64, Math.max(16, hidden1 / 2));

        int[] actorLayers = {stateSize, hidden1, hidden2, actionSize};
        this.actorNetwork = new ActorCriticNetwork(actorLayers, learningRate, false);

        int[] criticLayers = {stateSize, hidden1, hidden2, 1};
        this.criticNetwork = new ActorCriticNetwork(criticLayers, learningRate, true);

        this.trajectoryBuffer = new ArrayList<>();
        this.inputMean = new double[stateSize];
        this.inputM2 = new double[stateSize];

        System.out.println("[PPOAgent-IT] Initialized v1.1-IT with actor=" + Arrays.toString(actorLayers)
                + ", critic=" + Arrays.toString(criticLayers)
                + ", clip_eps=" + clipEpsilon + ", GAE_lambda=" + gaeLambda
                + ", epochs=" + ppoEpochs + ", batch=" + miniBatchSize
                + " [PPO-Clip strict-inequality fix enabled]");
    }

    /**
     * Standard constructor with sensible defaults for cybersecurity environments.
     */
    public PPOAgent(int stateSize, int actionSize, double learningRate, double discountFactor) {
        this(stateSize, actionSize, learningRate, discountFactor,
                0.2,    // clipEpsilon (PPO standard)
                0.95,   // gaeLambda
                4,      // ppoEpochs
                64,     // miniBatchSize
                0.01    // entropyCoeff
        );
    }

    // ==================== Agent Interface ====================

    @Override
    public Action selectAction(State state) {
        totalSteps++;

        updateInputStats(state.getFeatures());
        double[] normalizedState = normalizeInput(state.getFeatures());

        double[] logits = actorNetwork.forward(normalizedState);
        double[] probs = softmax(logits);

        double[] valueOut = criticNetwork.forward(normalizedState);
        double value = valueOut[0];

        int actionIndex = sampleFromDistribution(probs);

        double logProb = Math.log(Math.max(probs[actionIndex], 1e-10));

        trajectoryBuffer.add(new TrajectoryStep(
                state.getFeatures().clone(), actionIndex, logProb, value, probs.clone()
        ));

        return new Action(actionIndex, "ppo_action");
    }

    @Override
    public void updatePolicy(EpisodeResult episode) {
        episodeCount++;

        List<Transition> transitions = episode.getTransitions();
        int bufferStart = trajectoryBuffer.size() - transitions.size();

        if (bufferStart < 0) {
            trajectoryBuffer.clear();
            return;
        }

        for (int i = 0; i < transitions.size(); i++) {
            int bufIdx = bufferStart + i;
            if (bufIdx < trajectoryBuffer.size()) {
                double rawReward = transitions.get(i).reward;
                updateRewardStats(rawReward);
                trajectoryBuffer.get(bufIdx).reward = normalizeReward(rawReward);
                trajectoryBuffer.get(bufIdx).done = (i == transitions.size() - 1);
            }
        }

        computeGAE(bufferStart);

        List<TrajectoryStep> episodeSteps = new ArrayList<>(
                trajectoryBuffer.subList(bufferStart, trajectoryBuffer.size()));

        double totalPolicyLoss = 0;
        double totalValueLoss = 0;
        double totalEntropy = 0;
        int updateCount = 0;

        for (int epoch = 0; epoch < ppoEpochs; epoch++) {
            Collections.shuffle(episodeSteps, random);

            for (int batchStart = 0; batchStart < episodeSteps.size(); batchStart += miniBatchSize) {
                int batchEnd = Math.min(batchStart + miniBatchSize, episodeSteps.size());
                List<TrajectoryStep> miniBatch = episodeSteps.subList(batchStart, batchEnd);

                double[] losses = updateOnMiniBatch(miniBatch);
                totalPolicyLoss += losses[0];
                totalValueLoss += losses[1];
                totalEntropy += losses[2];
                updateCount++;
            }
        }

        if (updateCount > 0) {
            recentPolicyLoss = totalPolicyLoss / updateCount;
            recentValueLoss = totalValueLoss / updateCount;
            recentEntropy = totalEntropy / updateCount;
        }

        trajectoryBuffer.clear();

        updateLearningRate();

        entropyCoeff = Math.max(0.001, entropyCoeff * 0.999);
    }

    // ==================== GAE Computation ====================

    private void computeGAE(int startIdx) {
        double lastGAE = 0.0;
        int endIdx = trajectoryBuffer.size();

        for (int t = endIdx - 1; t >= startIdx; t--) {
            TrajectoryStep step = trajectoryBuffer.get(t);

            double nextValue;
            if (step.done || t == endIdx - 1) {
                nextValue = 0.0;
            } else {
                nextValue = trajectoryBuffer.get(t + 1).value;
            }

            double delta = step.reward + discountFactor * nextValue - step.value;

            if (step.done) {
                lastGAE = delta;
            } else {
                lastGAE = delta + discountFactor * gaeLambda * lastGAE;
            }

            step.advantage = lastGAE;
            step.returnValue = lastGAE + step.value;
        }

        double meanAdv = 0, m2Adv = 0;
        int count = 0;
        for (int t = startIdx; t < endIdx; t++) {
            count++;
            double delta = trajectoryBuffer.get(t).advantage - meanAdv;
            meanAdv += delta / count;
            m2Adv += delta * (trajectoryBuffer.get(t).advantage - meanAdv);
        }
        double stdAdv = count > 1 ? Math.sqrt(m2Adv / (count - 1) + 1e-8) : 1.0;

        for (int t = startIdx; t < endIdx; t++) {
            trajectoryBuffer.get(t).advantage =
                    (trajectoryBuffer.get(t).advantage - meanAdv) / stdAdv;
            trajectoryBuffer.get(t).advantage =
                    Math.max(-5.0, Math.min(5.0, trajectoryBuffer.get(t).advantage));
        }
    }

    // ==================== PPO Mini-Batch Update ====================

    /**
     * PPO-Clip update on a mini-batch.
     *
     * @return [policyLoss, valueLoss, entropy]
     */
    private double[] updateOnMiniBatch(List<TrajectoryStep> miniBatch) {
        double batchPolicyLoss = 0;
        double batchValueLoss = 0;
        double batchEntropy = 0;

        for (TrajectoryStep step : miniBatch) {
            double[] normalizedState = normalizeInput(step.state);

            // === Actor Update (PPO-Clip) ===
            double[] logits = actorNetwork.forward(normalizedState);
            double[] newProbs = softmax(logits);
            double newLogProb = Math.log(Math.max(newProbs[step.action], 1e-10));

            double ratio = Math.exp(newLogProb - step.oldLogProb);

            double surr1 = ratio * step.advantage;
            double surr2 = Math.max(1.0 - clipEpsilon, Math.min(1.0 + clipEpsilon, ratio))
                    * step.advantage;
            double policyLoss = -Math.min(surr1, surr2);

            // Branch-frequency counters (once per step; relationship surr1 vs surr2
            // is independent of action index a).
            if (surr1 < surr2) {
                diagSurrLess++;
            } else if (surr1 == surr2) {
                diagSurrEqual++;
            } else {
                diagSurrGreater++;
            }

            // Entropy: H(pi) = -sum_a p(a) * log(p(a))
            double entropy = 0;
            for (int a = 0; a < actionSize; a++) {
                if (newProbs[a] > 1e-10) {
                    entropy -= newProbs[a] * Math.log(newProbs[a]);
                }
            }

            // Entropy gradient via softmax Jacobian
            double[] entropyGrad = new double[actionSize];
            for (int a = 0; a < actionSize; a++) {
                double grad = 0.0;
                for (int b = 0; b < actionSize; b++) {
                    double indicator = (a == b) ? 1.0 : 0.0;
                    double jacobian = newProbs[b] * (indicator - newProbs[a]);
                    double dH_dp = -Math.log(Math.max(newProbs[b], 1e-10)) - 1.0;
                    grad += jacobian * dH_dp;
                }
                entropyGrad[a] = grad;
            }

            double[] actorTargets = new double[actionSize];
            for (int a = 0; a < actionSize; a++) {
                double oneHot = (a == step.action) ? 1.0 : 0.0;

                double policyGrad;
                if (surr1 <= surr2) {
                    // Unclipped (or unclipped-boundary): full PPO gradient with importance ratio
                    policyGrad = ratio * step.advantage * (oneHot - newProbs[a]);
                } else {
                    // Genuinely clipped (A>0 & r>1+eps, OR A<0 & r<1-eps): zero gradient
                    policyGrad = 0;
                }

                actorTargets[a] = logits[a] + (policyGrad + entropyCoeff * entropyGrad[a]);
            }
            actorNetwork.train(normalizedState, actorTargets);

            // === Critic Update ===
            double[] valueOut = criticNetwork.forward(normalizedState);
            double valuePred = valueOut[0];
            double valueError = valuePred - step.returnValue;
            double valueLoss = 0.5 * valueError * valueError;

            double[] criticTarget = {step.returnValue};
            criticNetwork.train(normalizedState, criticTarget);

            batchPolicyLoss += policyLoss;
            batchValueLoss += valueLoss;
            batchEntropy += entropy;
        }

        int n = Math.max(1, miniBatch.size());
        return new double[]{batchPolicyLoss / n, batchValueLoss / n, batchEntropy / n};
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

    // ==================== Learning Rate Annealing ====================

    private void updateLearningRate() {
        double progress = Math.min(1.0, (double) episodeCount / 1000.0);
        learningRate = initialLearningRate * (1.0 - 0.9 * progress);
        learningRate = Math.max(initialLearningRate * 0.1, learningRate);
        actorNetwork.setLearningRate(learningRate);
        criticNetwork.setLearningRate(learningRate);
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
        System.out.println("[PPOAgent-IT] Saving model to: " + path);
    }

    @Override
    public void loadModel(String path) {
        System.out.println("[PPOAgent-IT] Loading model from: " + path);
    }

    public AgentDTO toDTO(String agentId, String environment) {
        Map<String, Object> config = new HashMap<>();
        config.put("stateSize", stateSize);
        config.put("actionSize", actionSize);
        config.put("learningRate", learningRate);
        config.put("initialLearningRate", initialLearningRate);
        config.put("discountFactor", discountFactor);
        config.put("clipEpsilon", clipEpsilon);
        config.put("gaeLambda", gaeLambda);
        config.put("ppoEpochs", ppoEpochs);
        config.put("miniBatchSize", miniBatchSize);
        config.put("entropyCoeff", entropyCoeff);
        config.put("episodeCount", episodeCount);
        config.put("rewardMean", rewardMean);
        config.put("rewardM2", rewardM2);
        config.put("rewardCount", rewardCount);
        config.put("itDiagSurrLess", diagSurrLess);
        config.put("itDiagSurrEqual", diagSurrEqual);
        config.put("itDiagSurrGreater", diagSurrGreater);

        return new AgentDTO.Builder("PPO-IT", environment)
                .withAgentId(agentId)
                .withConfig(config)
                .withStatistics(episodeCount, recentPolicyLoss, recentValueLoss)
                .build();
    }

    public static PPOAgent fromDTO(AgentDTO dto) {
        Map<String, Object> config = dto.config;
        int stateSize = ((Number) config.get("stateSize")).intValue();
        int actionSize = ((Number) config.get("actionSize")).intValue();
        double learningRate = ((Number) config.get("learningRate")).doubleValue();
        double discountFactor = ((Number) config.get("discountFactor")).doubleValue();

        double clipEps = config.containsKey("clipEpsilon") ?
                ((Number) config.get("clipEpsilon")).doubleValue() : 0.2;
        double gaeL = config.containsKey("gaeLambda") ?
                ((Number) config.get("gaeLambda")).doubleValue() : 0.95;
        int epochs = config.containsKey("ppoEpochs") ?
                ((Number) config.get("ppoEpochs")).intValue() : 4;
        int batch = config.containsKey("miniBatchSize") ?
                ((Number) config.get("miniBatchSize")).intValue() : 64;
        double entropy = config.containsKey("entropyCoeff") ?
                ((Number) config.get("entropyCoeff")).doubleValue() : 0.01;

        PPOAgent agent = new PPOAgent(
                stateSize, actionSize, learningRate, discountFactor,
                clipEps, gaeL, epochs, batch, entropy);

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
    public double getLearningRate() { return learningRate; }
    public double getDiscountFactor() { return discountFactor; }
    public double getClipEpsilon() { return clipEpsilon; }
    public double getGaeLambda() { return gaeLambda; }
    public int getEpisodeCount() { return episodeCount; }
    public int getTotalSteps() { return totalSteps; }
    public double getRecentPolicyLoss() { return recentPolicyLoss; }
    public double getRecentValueLoss() { return recentValueLoss; }
    public double getRecentEntropy() { return recentEntropy; }
    public double getRewardMean() { return rewardMean; }
    public double getRewardStd() {
        return rewardCount > 1 ? Math.sqrt(rewardM2 / (rewardCount - 1)) : 0;
    }
    public double getEntropyCoeff() { return entropyCoeff; }

    // ==================== IT-Diag Getters ====================

    public long getDiagSurrLess() { return diagSurrLess; }
    public long getDiagSurrEqual() { return diagSurrEqual; }
    public long getDiagSurrGreater() { return diagSurrGreater; }
    public long getDiagSurrTotal() { return diagSurrLess + diagSurrEqual + diagSurrGreater; }

    public void resetDiagCounters() {
        diagSurrLess = 0;
        diagSurrEqual = 0;
        diagSurrGreater = 0;
    }

    /**
     * Human-readable summary of branch-frequency counters.
     * "equal" is the bug region: in publication PPOAgent.java these cases were
     * incorrectly routed to zero gradient.
     */
    public String getDiagSummary() {
        long total = diagSurrLess + diagSurrEqual + diagSurrGreater;
        if (total == 0) return "no PPO updates yet";
        return String.format(Locale.US,
                "surr1<surr2: %,d (%.2f%%) | surr1==surr2 [bug region]: %,d (%.2f%%) | surr1>surr2: %,d (%.2f%%)",
                diagSurrLess, 100.0 * diagSurrLess / total,
                diagSurrEqual, 100.0 * diagSurrEqual / total,
                diagSurrGreater, 100.0 * diagSurrGreater / total);
    }

    // ==================== Trajectory Step ====================

    private static class TrajectoryStep {
        final double[] state;
        final int action;
        final double oldLogProb;
        double value;
        final double[] oldProbs;
        double reward;
        boolean done;
        double advantage;
        double returnValue;

        TrajectoryStep(double[] state, int action, double logProb,
                       double value, double[] probs) {
            this.state = state;
            this.action = action;
            this.oldLogProb = logProb;
            this.value = value;
            this.oldProbs = probs;
            this.reward = 0;
            this.done = false;
            this.advantage = 0;
            this.returnValue = 0;
        }
    }

    // ==================== Actor-Critic Network ====================

    /**
     * Fully-connected feed-forward network used as both actor (action logits)
     * and critic (state-value scalar). Leaky-ReLU hidden activations, linear
     * output, He initialization, element-wise gradient clipping.
     */
    static class ActorCriticNetwork {
        private final int[] layerSizes;
        double[][] weights;
        double[][] biases;
        private double learningRate;
        private final Random random;

        private double[][] activations;
        private double[][] preActivations;

        private static final double LEAKY_ALPHA = 0.01;
        private static final double GRADIENT_CLIP = 1.0;
        private final boolean isCritic;

        ActorCriticNetwork(int[] layers, double learningRate, boolean isCritic) {
            this.layerSizes = layers.clone();
            this.learningRate = learningRate;
            this.isCritic = isCritic;
            this.random = new Random(42);

            int numLayers = layers.length;
            weights = new double[numLayers - 1][];
            biases = new double[numLayers - 1][];

            for (int i = 0; i < numLayers - 1; i++) {
                int inputSize = layers[i];
                int outputSize = layers[i + 1];

                double scale = Math.sqrt(2.0 / inputSize);
                weights[i] = new double[inputSize * outputSize];
                for (int j = 0; j < weights[i].length; j++) {
                    weights[i][j] = random.nextGaussian() * scale;
                }

                biases[i] = new double[outputSize];
                if (i == numLayers - 2 && isCritic) {
                    Arrays.fill(biases[i], 0.0);
                } else {
                    for (int j = 0; j < biases[i].length; j++) {
                        biases[i][j] = 0.01;
                    }
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

        void setLearningRate(double lr) { this.learningRate = lr; }
        int[] getLayerSizes() { return layerSizes.clone(); }
    }
}
