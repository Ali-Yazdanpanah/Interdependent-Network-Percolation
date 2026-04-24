#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <queue>
#include <random>
#include <vector>

using Graph = std::vector<std::vector<int>>;

Graph generate_ER_graph(int n, double mean_k, std::mt19937_64& rng) {
    Graph graph(n);
    if (n <= 1) {
        return graph;
    }

    const double p_edge = mean_k / static_cast<double>(n - 1);
    std::bernoulli_distribution edge_dist(p_edge);

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (edge_dist(rng)) {
                graph[i].push_back(j);
                graph[j].push_back(i);
            }
        }
    }
    return graph;
}

std::vector<bool> get_LCC(const Graph& graph, const std::vector<bool>& active_nodes) {
    const int n = static_cast<int>(graph.size());
    std::vector<bool> visited(n, false);
    std::vector<int> queue_storage;
    queue_storage.reserve(n);

    std::vector<int> best_component;
    best_component.reserve(n);
    std::vector<int> current_component;
    current_component.reserve(n);

    for (int start = 0; start < n; ++start) {
        if (!active_nodes[start] || visited[start]) {
            continue;
        }

        current_component.clear();
        queue_storage.clear();
        queue_storage.push_back(start);
        visited[start] = true;

        for (std::size_t head = 0; head < queue_storage.size(); ++head) {
            const int node = queue_storage[head];
            current_component.push_back(node);
            for (const int nbr : graph[node]) {
                if (active_nodes[nbr] && !visited[nbr]) {
                    visited[nbr] = true;
                    queue_storage.push_back(nbr);
                }
            }
        }

        if (current_component.size() > best_component.size()) {
            best_component = current_component;
        }
    }

    std::vector<bool> lcc_nodes(n, false);
    for (const int node : best_component) {
        lcc_nodes[node] = true;
    }
    return lcc_nodes;
}

double interdependent_percolation(
    const Graph& G_A,
    const Graph& G_B,
    double p,
    std::mt19937_64& rng
) {
    const int n = static_cast<int>(G_A.size());
    if (n == 0) {
        return 0.0;
    }

    std::vector<bool> active_A(n, true);
    std::vector<bool> active_B(n, true);

    // Initial trigger: remove a fraction p from network A.
    int remove_count = static_cast<int>(std::llround(p * n));
    remove_count = std::clamp(remove_count, 0, n);
    std::vector<int> nodes(n);
    std::iota(nodes.begin(), nodes.end(), 0);
    std::shuffle(nodes.begin(), nodes.end(), rng);
    for (int i = 0; i < remove_count; ++i) {
        active_A[nodes[i]] = false;
    }

    while (true) {
        const int prev_A = static_cast<int>(std::count(active_A.begin(), active_A.end(), true));
        const int prev_B = static_cast<int>(std::count(active_B.begin(), active_B.end(), true));

        // a) Dependency failure in B.
        for (int i = 0; i < n; ++i) {
            active_B[i] = active_B[i] && active_A[i];
        }

        // b) Connectivity failure in B.
        active_B = get_LCC(G_B, active_B);

        // c) Dependency failure in A.
        for (int i = 0; i < n; ++i) {
            active_A[i] = active_A[i] && active_B[i];
        }

        // d) Connectivity failure in A.
        active_A = get_LCC(G_A, active_A);

        const int curr_A = static_cast<int>(std::count(active_A.begin(), active_A.end(), true));
        const int curr_B = static_cast<int>(std::count(active_B.begin(), active_B.end(), true));
        if (curr_A == prev_A && curr_B == prev_B) {
            return static_cast<double>(curr_A) / static_cast<double>(n);
        }
    }
}

int main() {
    constexpr int n = 1000;
    constexpr double k = 4.0;
    constexpr int mc_runs = 10;
    constexpr double p_step = 0.02;

    std::random_device rd;
    std::mt19937_64 rng(rd());

    const Graph G_A = generate_ER_graph(n, k, rng);
    const Graph G_B = generate_ER_graph(n, k, rng);

    std::cout << "p,average_LCC_fraction\n";
    const int steps = static_cast<int>(1.0 / p_step);
    for (int i = 0; i <= steps; ++i) {
        const double p = i * p_step;
        double sum = 0.0;
        for (int run = 0; run < mc_runs; ++run) {
            sum += interdependent_percolation(G_A, G_B, p, rng);
        }
        const double average = sum / static_cast<double>(mc_runs);
        std::cout << std::fixed << std::setprecision(2) << p << ","
                  << std::setprecision(6) << average << "\n";
    }

    return 0;
}
