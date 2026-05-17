#include <iostream>
#include <vector>
#include <queue>
#include <omp.h>

using namespace std;

class Graph {

    int V;
    vector<vector<int>> adj;
    vector<bool> visitedDFS;

public:

    // Constructor
    Graph(int V) {
        this->V = V;
        adj.resize(V);
        visitedDFS.resize(V, false);
    }

    // Add edge to graph
    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    // ---------------- PARALLEL BFS ----------------
    void parallelBFS(int start) {

        vector<bool> visited(V, false);
        queue<int> q;

        visited[start] = true;
        q.push(start);

        cout << "Parallel BFS Traversal: ";

        while (!q.empty()) {

            int node = -1;

            // Critical section for queue access
            #pragma omp critical
            {
                if (!q.empty()) {
                    node = q.front();
                    q.pop();
                    cout << node << " ";
                }
            }

            if (node == -1)
                continue;

            // Parallel processing of neighbors
            #pragma omp parallel for
            for (int i = 0; i < adj[node].size(); i++) {

                int neighbor = adj[node][i];

                if (!visited[neighbor]) {

                    #pragma omp critical
                    {
                        if (!visited[neighbor]) {
                            visited[neighbor] = true;
                            q.push(neighbor);
                        }
                    }
                }
            }
        }

        cout << endl;
    }

    // ---------------- PARALLEL DFS UTILITY ----------------
    void parallelDFSUtil(int node) {

        bool alreadyVisited = false;

        // Critical section for visited array
        #pragma omp critical
        {
            if (visitedDFS[node]) {
                alreadyVisited = true;
            }
            else {
                visitedDFS[node] = true;
                cout << node << " ";
            }
        }

        if (alreadyVisited)
            return;

        // Create parallel tasks for neighbors
        for (int neighbor : adj[node]) {

            #pragma omp task
            {
                if (!visitedDFS[neighbor]) {
                    parallelDFSUtil(neighbor);
                }
            }
        }
    }

    // ---------------- PARALLEL DFS ----------------
    void parallelDFS(int start) {

        cout << "Parallel DFS Traversal: ";

        #pragma omp parallel
        {
            #pragma omp single
            {
                parallelDFSUtil(start);
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
