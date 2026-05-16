//Parallel BFS_DFS using OpenMP
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
    Graph g(6);
    // Creating graph
    g.addEdge(0, 1);
    g.addEdge(0, 2);
    g.addEdge(1, 3);
    g.addEdge(1, 4);
    g.addEdge(2, 5);
    // Perform Parallel BFS
    g.parallelBFS(0);
    // Perform Parallel DFS
    g.parallelDFS(0);
    return 0;
}
/*
Graph Structure
Your graph is:
        0
      /   \
     1     2
    / \     \
   3   4     5
BFS Traversal
BFS visits level by level:
1.	Start at 0 
2.	Visit neighbors → 1 2 
3.	Visit next level → 3 4 5 
So common BFS output:
0 1 2 3 4 5
DFS Traversal
DFS order depends on:
•	Task scheduling 
•	Thread execution order
*/
