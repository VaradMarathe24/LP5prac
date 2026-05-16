#include <iostream>
#include <omp.h>
#include <climits>
#include <cstdlib>
#include <ctime>
using namespace std;

#define N 100000   // large enough to show real parallel speedup

// ============================
// Parallel Min
// ============================
// reduction(min:minVal) gives each thread a private copy initialised
// to INT_MAX, then combines all copies with min() at the end.
int findMin(int arr[], int n) {
    int minVal = arr[0];
    #pragma omp parallel for reduction(min:minVal)
    for (int i = 0; i < n; i++)
        if (arr[i] < minVal) minVal = arr[i];
    return minVal;
}

// ============================
// Parallel Max
// ============================
int findMax(int arr[], int n) {
    int maxVal = arr[0];
    #pragma omp parallel for reduction(max:maxVal)
    for (int i = 0; i < n; i++)
        if (arr[i] > maxVal) maxVal = arr[i];
    return maxVal;
}

// ============================
// Parallel Sum
// ============================
// reduction(+:total) gives each thread a private total=0,
// then adds all private totals together at the end.
long long findSum(int arr[], int n) {
    long long total = 0;
    #pragma omp parallel for reduction(+:total)
    for (int i = 0; i < n; i++)
        total += arr[i];
    return total;
}

// ============================
// Average (reuses findSum)
// ============================
double findAverage(int arr[], int n) {
    return (double)findSum(arr, n) / n;
}

// ============================
// Main
// ============================
int main() {
    srand(time(0));

    // Generate large random array for meaningful timing
    int* arr = new int[N];
    for (int i = 0; i < N; i++)
        arr[i] = rand() % 100000 + 1;

    double start, end;

    // -------- Min --------
    start = omp_get_wtime();
    int minimum = findMin(arr, N);
    end   = omp_get_wtime();
    cout << "Minimum Value         = " << minimum << endl;
    cout << "Time for Min          = " << (end - start) << " seconds\n\n";

    // -------- Max --------
    start = omp_get_wtime();
    int maximum = findMax(arr, N);
    end   = omp_get_wtime();
    cout << "Maximum Value         = " << maximum << endl;
    cout << "Time for Max          = " << (end - start) << " seconds\n\n";

    // -------- Sum --------
    start = omp_get_wtime();
    long long total = findSum(arr, N);
    end   = omp_get_wtime();
    cout << "Sum                   = " << total << endl;
    cout << "Time for Sum          = " << (end - start) << " seconds\n\n";

    // -------- Average --------
    start = omp_get_wtime();
    double avg = findAverage(arr, N);
    end   = omp_get_wtime();
    cout << "Average               = " << avg << endl;
    cout << "Time for Average      = " << (end - start) << " seconds\n\n";

    delete[] arr;
    return 0;
}

/*
 * Compile:
 *   g++ -fopenmp -O2 -o reduction reduction.cpp
 *
 * How reduction works (examiner question):
 *   1. OpenMP splits the loop across T threads.
 *   2. Each thread has its OWN private copy of the reduction variable
 *      (initialised to identity: 0 for +, INT_MAX for min, INT_MIN for max).
 *   3. Each thread updates only its private copy — no race condition.
 *   4. After the loop, OpenMP combines all private copies using the
 *      reduction operator and stores the final result in the original variable.
 */
