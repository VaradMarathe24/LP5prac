#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>
using namespace std;

class Graph {
    int V;
    vector<vector<int>> adj;
public:
    Graph(int V) : V(V), adj(V) {}

    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u); // undirected graph
    }

    // â”€â”€â”€ Parallel BFS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    // FIX 1: Snapshot the entire current level into a local vector BEFORE
    //         spawning threads.  This eliminates the race on q.front()/pop().
    // FIX 2: Print AFTER the parallel region so output is deterministic.
    void parallelBFS(int start) {
        vector<bool> visited(V, false);
        queue<int> q;
        visited[start] = true;
        q.push(start);

        cout << "\nParallel BFS Traversal: ";

        while (!q.empty()) {
            // --- Drain the current level into a snapshot (serial, fast) ---
            vector<int> level;
            while (!q.empty()) {
                level.push_back(q.front());
                q.pop();
            }

            // --- Process all nodes in this level in parallel ---
            #pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < (int)level.size(); i++) {
                int node = level[i];
                for (int neighbor : adj[node]) {
                    if (!visited[neighbor]) {          // unsync read: fast filter
                        #pragma omp critical
                        {
                            if (!visited[neighbor]) {  // sync re-check: correct
                                visited[neighbor] = true;
                                q.push(neighbor);
                            }
                        }
                    }
                }
            }

            // Print level results (serial, ordered)
            for (int node : level) cout << node << " ";
        }
        cout << endl;
    }

    // â”€â”€â”€ Parallel DFS (OpenMP task tree) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    // FIX 3: Remove nested "#pragma omp parallel for" from the utility.
    //         Tasks must belong to the single parallel region created in
    //         parallelDFS(); creating extra teams inside tasks is illegal here.
    // FIX 4: Remove the unsynchronised read of visited[] before task spawn;
    //         let the critical block inside the utility handle deduplication.
    void parallelDFSUtil(int node, vector<bool> &visited) {
        bool alreadyVisited;
        #pragma omp critical
        {
            alreadyVisited = visited[node];
            if (!visited[node]) {
                visited[node] = true;
                cout << node << " ";
            }
        }
        if (alreadyVisited) return;

        // Spawn one task per unvisited neighbour.
        // The critical block above guarantees each node is printed exactly once.
        for (int i = 0; i < (int)adj[node].size(); i++) {
            int neighbor = adj[node][i];
            #pragma omp task firstprivate(neighbor)
            parallelDFSUtil(neighbor, visited);
        }
        // Wait for all child tasks before this task returns.
        #pragma omp taskwait
    }

    void parallelDFS(int start) {
        vector<bool> visited(V, false);
        cout << "\nParallel DFS Traversal: ";
        #pragma omp parallel
        {
            #pragma omp single nowait
            {
                parallelDFSUtil(start, visited);
            }
        }
        cout << endl;
    }
};

int main() {
    int V, E;
    cout << "Enter number of vertices: ";
    cin >> V;
    Graph g(V);

    cout << "Enter number of edges: ";
    cin >> E;

    cout << "Enter edges (u v):\n";
    for (int i = 0; i < E; i++) {
        int u, v;
        cin >> u >> v;
        g.addEdge(u, v);
    }

    int start;
    cout << "Enter starting vertex: ";
    cin >> start;

    g.parallelBFS(start);
    g.parallelDFS(start);

    return 0;
}

/*
 * Sample tree used for testing:
 *
 *        0
 *       / \
 *      1   2
 *     / \   \
 *    3   4   5
 *
 * Input:
 *   Vertices: 6,  Edges: 5
 *   0 1 | 0 2 | 1 3 | 1 4 | 2 5
 *   Start: 0
 *
 * Expected BFS output (level-order):  0 1 2 3 4 5
 * Expected DFS output (pre-order):    0 1 3 4 2 5
 *   (task scheduling may permute siblings, e.g. 0 2 5 1 3 4)
 *
 * Compile:
 *   g++ -fopenmp -O2 -o pbfs_dfs parallel_bfs_dfs.cpp
 */
