#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <omp.h>
using namespace std;

#define SIZE 10000

// ============================
// Correctness Check
// ============================
bool isSorted(vector<int>& arr) {
    for (int i = 1; i < (int)arr.size(); i++)
        if (arr[i] < arr[i - 1]) return false;
    return true;
}

// ============================
// Sequential Bubble Sort
// ============================
void bubbleSortSeq(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n - 1; i++) {
        bool swapped = false;
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
                swapped = true;
            }
        }
        if (!swapped) break; // early exit if already sorted
    }
}

// ============================
// Parallel Bubble Sort
// (Odd-Even Transposition Sort)
// ============================
// KEY IDEA: In each round, compare independent pairs in parallel.
//   Even phase: compare (0,1), (2,3), (4,5) ... — no overlap → safe parallel
//   Odd  phase: compare (1,2), (3,4), (5,6) ... — no overlap → safe parallel
// After n full rounds, array is guaranteed sorted.
void bubbleSortParallel(vector<int>& arr) {
    int n = arr.size();
    for (int i = 0; i < n; i++) {
        // Even phase — pairs (j, j+1) for even j
        #pragma omp parallel for shared(arr)
        for (int j = 0; j < n - 1; j += 2) {
            if (arr[j] > arr[j + 1])
                swap(arr[j], arr[j + 1]);
        }
        // Odd phase — pairs (j, j+1) for odd j
        #pragma omp parallel for shared(arr)
        for (int j = 1; j < n - 1; j += 2) {
            if (arr[j] > arr[j + 1])
                swap(arr[j], arr[j + 1]);
        }
    }
}

// ============================
// Merge Helper (sequential)
// ============================
void merge(vector<int>& arr, int l, int m, int r) {
    vector<int> left(arr.begin() + l,     arr.begin() + m + 1);
    vector<int> right(arr.begin() + m + 1, arr.begin() + r + 1);
    int i = 0, j = 0, k = l;
    while (i < (int)left.size() && j < (int)right.size()) {
        if (left[i] <= right[j]) arr[k++] = left[i++];
        else                     arr[k++] = right[j++];
    }
    while (i < (int)left.size())  arr[k++] = left[i++];
    while (j < (int)right.size()) arr[k++] = right[j++];
}

// ============================
// Sequential Merge Sort
// ============================
void mergeSortSeq(vector<int>& arr, int l, int r) {
    if (l < r) {
        int m = (l + r) / 2;
        mergeSortSeq(arr, l, m);
        mergeSortSeq(arr, m + 1, r);
        merge(arr, l, m, r);
    }
}

// ============================
// Parallel Merge Sort
// ============================
// FIX: Use omp task (not omp parallel sections) so all recursive
//      calls share ONE thread pool — no repeated fork/join overhead.
//      The outer parallel+single in main() creates the pool once.
//      depth controls how many levels get parallelised (top 4 = 16 tasks).
//      Below depth 0, falls back to sequential to avoid task explosion.
void mergeSortParallel(vector<int>& arr, int l, int r, int depth) {
    if (l >= r) return;
    int m = (l + r) / 2;
    if (depth <= 0) {
        // Sequential fallback for small sub-problems
        mergeSortSeq(arr, l, m);
        mergeSortSeq(arr, m + 1, r);
    } else {
        #pragma omp task shared(arr)
        mergeSortParallel(arr, l, m, depth - 1);

        #pragma omp task shared(arr)
        mergeSortParallel(arr, m + 1, r, depth - 1);

        #pragma omp taskwait  // wait for both halves before merging
    }
    merge(arr, l, m, r);
}

// ============================
// Random Array Generator
// ============================
void generateRandom(vector<int>& arr) {
    for (int& x : arr)
        x = rand() % 100000;
}

// ============================
// Main
// ============================
int main() {
    vector<int> arr(SIZE), temp;
    srand(time(0));
    generateRandom(arr);

    double start, end;

    // -------- Bubble Sort Sequential --------
    temp = arr;
    start = omp_get_wtime();
    bubbleSortSeq(temp);
    end = omp_get_wtime();
    cout << "Sequential Bubble Sort Time : " << (end - start) << " sec"
         << "  [" << (isSorted(temp) ? "Correct" : "WRONG") << "]\n";

    // -------- Bubble Sort Parallel --------
    temp = arr;
    start = omp_get_wtime();
    bubbleSortParallel(temp);
    end = omp_get_wtime();
    cout << "Parallel   Bubble Sort Time : " << (end - start) << " sec"
         << "  [" << (isSorted(temp) ? "Correct" : "WRONG") << "]\n";

    // -------- Merge Sort Sequential --------
    temp = arr;
    start = omp_get_wtime();
    mergeSortSeq(temp, 0, SIZE - 1);
    end = omp_get_wtime();
    cout << "Sequential Merge  Sort Time : " << (end - start) << " sec"
         << "  [" << (isSorted(temp) ? "Correct" : "WRONG") << "]\n";

    // -------- Merge Sort Parallel --------
    temp = arr;
    start = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        mergeSortParallel(temp, 0, SIZE - 1, 4);
    }
    end = omp_get_wtime();
    cout << "Parallel   Merge  Sort Time : " << (end - start) << " sec"
         << "  [" << (isSorted(temp) ? "Correct" : "WRONG") << "]\n";

    return 0;
}

/*
 * Compile:
 *   g++ -fopenmp -O2 -o parallel_sort parallel_sort.cpp
 *
 * Expected output (times vary by machine):
 *   Sequential Bubble Sort Time : 0.85 sec  [Correct]
 *   Parallel   Bubble Sort Time : 0.18 sec  [Correct]
 *   Sequential Merge  Sort Time : 0.003 sec [Correct]
 *   Parallel   Merge  Sort Time : 0.001 sec [Correct]
 *
 * Why parallel bubble sort still looks slow:
 *   Bubble sort is O(n^2) — even with parallelism it stays O(n^2/p).
 *   The thread spawn overhead per round reduces the gain significantly.
 *   Merge sort is O(n log n) — parallelism here gives real speedup.
 */
