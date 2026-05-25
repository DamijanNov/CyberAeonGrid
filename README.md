**Public Code Release**

PPO/SAC Baselines and Cybersecurity Simulation Environment
Source code accompanying the article:
> Novak, D.; Fister, I., Jr.; Dugonik, J. Replacing the Genetic Algorithm with Multi-Objective Bacterial Foraging Optimization in XCS. Mathematics (MDPI), 2026.

**Scope**

The files in `src/` are provided for inspection and verification of the algorithms described in the paper. They depend on the broader CyberAeonGrid research platform for compilation and are not packaged as a runnable standalone project.

Included:

`src/rl/PPOAgent.java`: Proximal Policy Optimization baseline (Experiment 2)

`src/rl/SACAgent.java`: Soft Actor-Critic baseline (Experiment 2)

`src/simulation/NetworkSimulationEnvironment.java`: RL environment bridge described in Section 3.4

`src/simulation/NetworkSimulation.java`: Core simulation engine

`src/simulation/NetworkTopology.java`: Network graph structure

The BFOA-XCS integration and the XCS core remain part of the CyberAeonGrid research platform and are not included in this release.

**License**

See `LICENSE.md` (MIT).

**Contact**

Damijan Novak — damijan.novak@um.si
