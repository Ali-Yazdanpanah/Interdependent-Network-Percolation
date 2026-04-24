#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

using Graph = std::vector<std::vector<int>>;

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
    std::uint64_t seed
) {
    const int n = static_cast<int>(G_A.size());
    if (n == 0 || static_cast<int>(G_B.size()) != n) {
        return 0.0;
    }

    std::mt19937_64 rng(seed);

    std::vector<bool> active_A(n, true);
    std::vector<bool> active_B(n, true);

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

        for (int i = 0; i < n; ++i) {
            active_B[i] = active_B[i] && active_A[i];
        }
        active_B = get_LCC(G_B, active_B);

        for (int i = 0; i < n; ++i) {
            active_A[i] = active_A[i] && active_B[i];
        }
        active_A = get_LCC(G_A, active_A);

        const int curr_A = static_cast<int>(std::count(active_A.begin(), active_A.end(), true));
        const int curr_B = static_cast<int>(std::count(active_B.begin(), active_B.end(), true));
        if (curr_A == prev_A && curr_B == prev_B) {
            return static_cast<double>(curr_A) / static_cast<double>(n);
        }
    }
}

PYBIND11_MODULE(cascade_sim, m) {
    m.doc() = "Interdependent cascading failure simulator";
    m.def(
        "interdependent_percolation",
        &interdependent_percolation,
        pybind11::arg("G_A"),
        pybind11::arg("G_B"),
        pybind11::arg("p"),
        pybind11::arg("seed")
    );
}
