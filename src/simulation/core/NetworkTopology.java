package simulation.core;

import java.awt.*;
import java.util.List;
import java.util.*;
import java.util.stream.Collectors;

public class NetworkTopology {
    private List<NetworkNode> nodes;
    private List<NetworkEdge> edges;
    private Map<String, NetworkNode> nodeMap;

    public NetworkTopology() {
        this.nodes = new ArrayList<>();
        this.edges = new ArrayList<>();
        this.nodeMap = new HashMap<>();
    }

    // Reset topology
    public void reset() {
        for (NetworkNode node : nodes) {
            node.setCompromised(false);
            node.setDefended(false);
            node.setIsolated(false);
            node.setOverloaded(false);
            node.setAvailable(true);
            node.setPerformanceImpact(0);
            node.setEncrypted(false);
        }
    }

    // Generate topology with specific parameters
    public void generateTopology(int nodeCount, int edgeCount, int seed) {
        Random rand = new Random(seed);
        nodes.clear();
        edges.clear();
        nodeMap.clear();

        // Create nodes
        for (int i = 0; i < nodeCount; i++) {
            NetworkNode node = new NetworkNode("node_" + i);
            node.setType(NodeType.values()[rand.nextInt(NodeType.values().length)]);
            node.setPosition(new Point(rand.nextInt(800), rand.nextInt(600)));
            addNode(node);
        }

        // Create edges
        for (int i = 0; i < edgeCount; i++) {
            NetworkNode src = nodes.get(rand.nextInt(nodes.size()));
            NetworkNode dst = nodes.get(rand.nextInt(nodes.size()));
            if (!src.equals(dst) && !hasEdge(src, dst)) {
                addEdge(new NetworkEdge(src, dst));
            }
        }
    }

    // Get random node
    public NetworkNode getRandomNode() {
        if (nodes.isEmpty()) return null;
        return nodes.get(new Random().nextInt(nodes.size()));
    }

    // Get connections as edges
    public List<NetworkEdge> getConnections() {
        return new ArrayList<>(edges);
    }

    // Extended NetworkNode class
    public static class NetworkNode {
        private String id;
        private NodeType type;
        private boolean compromised;
        private boolean defended;
        private boolean isolated;
        private boolean overloaded;
        private boolean available;
        private boolean monitored;
        private boolean encrypted;
        private double vulnerabilityScore;
        private double criticality;
        private double performanceImpact;
        private Point position;
        private Map<String, Object> attributes;
        private Set<String> tags;
        private List<String> vulnerabilities;
        private Color color;

        public NetworkNode(String id) {
            this.id = id;
            this.type = NodeType.WORKSTATION;
            this.compromised = false;
            this.defended = false;
            this.isolated = false;
            this.overloaded = false;
            this.available = true;
            this.monitored = false;
            this.encrypted = false;
            this.vulnerabilityScore = 0.5;
            this.criticality = 0.5;
            this.performanceImpact = 0.0;
            this.position = new Point(0, 0);
            this.attributes = new HashMap<>();
            this.tags = new HashSet<>();
            this.vulnerabilities = new ArrayList<>();
            this.color = Color.GREEN;
        }

        // Position methods
        public Point getPosition() { return position; }
        public void setPosition(Point position) { this.position = position; }

        // Criticality methods
        public double getCriticality() { return criticality; }
        public void setCriticality(double criticality) { this.criticality = criticality; }

        // Performance impact
        public double getPerformanceImpact() { return performanceImpact; }
        public void setPerformanceImpact(double impact) { this.performanceImpact = impact; }

        // Encryption status
        public boolean isEncrypted() { return encrypted; }
        public void setEncrypted(boolean encrypted) { this.encrypted = encrypted; }

        // Vulnerabilities
        public List<String> getVulnerabilities() { return vulnerabilities; }
        public void addVulnerability(String vuln) { vulnerabilities.add(vuln); }

        // Tag management
        public void addTag(String tag) { tags.add(tag); }
        public void removeTag(String tag) { tags.remove(tag); }
        public boolean hasTag(String tag) { return tags.contains(tag); }

        // All other existing getters and setters
        public String getId() { return id; }
        public NodeType getType() { return type; }
        public void setType(NodeType type) { this.type = type; }
        public boolean isCompromised() { return compromised; }
        public void setCompromised(boolean compromised) {
            this.compromised = compromised;
            updateColor();
        }
        public boolean isDefended() { return defended; }
        public void setDefended(boolean defended) { this.defended = defended; }
        public boolean isIsolated() { return isolated; }
        public void setIsolated(boolean isolated) {
            this.isolated = isolated;
            updateColor();
        }
        public boolean isOverloaded() { return overloaded; }
        public void setOverloaded(boolean overloaded) {
            this.overloaded = overloaded;
            updateColor();
        }
        public boolean isAvailable() { return available; }
        public void setAvailable(boolean available) {
            this.available = available;
            updateColor();
        }
        public boolean isMonitored() { return monitored; }
        public void setMonitored(boolean monitored) { this.monitored = monitored; }
        public double getVulnerabilityScore() { return vulnerabilityScore; }
        public void setVulnerabilityScore(double score) { this.vulnerabilityScore = score; }
        public Map<String, Object> getAttributes() { return attributes; }
        public Color getColor() { return color; }
        public void setColor(Color color) { this.color = color; }

        private void updateColor() {
            if (compromised) {
                color = Color.RED;
            } else if (isolated) {
                color = Color.GRAY;
            } else if (overloaded) {
                color = Color.ORANGE;
            } else if (!available) {
                color = Color.DARK_GRAY;
            } else if (defended) {
                color = Color.BLUE;
            } else {
                color = Color.GREEN;
            }
        }
    }

    public enum NodeType {
        SERVER, WORKSTATION, ROUTER, FIREWALL, IDS
    }

    // NetworkConnection for compatibility
    public static class NetworkConnection extends NetworkEdge {
        public NetworkConnection(NetworkNode source, NetworkNode target) {
            super(source, target);
        }

        // Alias methods for compatibility
        public NetworkNode getFrom() { return getSource(); }
        public NetworkNode getTo() { return getTarget(); }
        public double getBandwidth() { return getWeight() * 100; } // Convert weight to bandwidth
    }

    // NetworkEdge class
    public static class NetworkEdge {
        private NetworkNode source;
        private NetworkNode target;
        private double weight;

        public NetworkEdge(NetworkNode source, NetworkNode target) {
            this.source = source;
            this.target = target;
            this.weight = 1.0;
        }

        public NetworkNode getSource() { return source; }
        public NetworkNode getTarget() { return target; }
        public double getWeight() { return weight; }
        public void setWeight(double weight) { this.weight = weight; }
    }

    // All other methods
    public void addNode(NetworkNode node) {
        nodes.add(node);
        nodeMap.put(node.getId(), node);
    }

    public void addEdge(NetworkEdge edge) {
        edges.add(edge);
    }

    public boolean hasEdge(NetworkNode src, NetworkNode dst) {
        for (NetworkEdge edge : edges) {
            if ((edge.getSource().equals(src) && edge.getTarget().equals(dst)) ||
                    (edge.getSource().equals(dst) && edge.getTarget().equals(src))) {
                return true;
            }
        }
        return false;
    }

    public Map<String, NetworkNode> getNodeMap() { return nodeMap; }
    public List<NetworkNode> getNodes() { return nodes; }
    public List<NetworkEdge> getEdges() { return edges; }
    public int getNodeCount() { return nodes.size(); }
    public int getEdgeCount() { return edges.size(); }

    public List<NetworkNode> getNeighbors(NetworkNode node) {
        List<NetworkNode> neighbors = new ArrayList<>();
        for (NetworkEdge edge : edges) {
            if (edge.getSource().equals(node)) {
                neighbors.add(edge.getTarget());
            } else if (edge.getTarget().equals(node)) {
                neighbors.add(edge.getSource());
            }
        }
        return neighbors;
    }

    public void generateRandomTopology(int nodeCount) {
        generateTopology(nodeCount, nodeCount * 2, (int) System.currentTimeMillis());
    }

    public List<NetworkNode> getCompromisedNodes() {
        return nodes.stream()
                .filter(NetworkNode::isCompromised)
                .collect(Collectors.toList());
    }

    public double getCompromisedRatio() {
        if (nodes.isEmpty()) return 0;
        return (double) getCompromisedNodes().size() / nodes.size();
    }

    public double getDefendedRatio() {
        if (nodes.isEmpty()) return 0;
        long defended = nodes.stream().filter(NetworkNode::isDefended).count();
        return (double) defended / nodes.size();
    }
}