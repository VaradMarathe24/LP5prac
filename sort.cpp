#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

/*========================================================
                PARALLEL BUBBLE SORT
==========================================================*/

/*
Bubble Sort:
- Uses Odd-Even Transposition Method
- Parallelized using OpenMP
- Time Complexity: O(n^2)
*/

void parallelBubbleSort(vector<int>& arr) {

    int n = arr.size();

    for (int i = 0; i < n; i++) {

        /*
        Odd-Even Parallel Phase

        i % 2 == 0  -> Even phase
        i % 2 == 1  -> Odd phase
        */

        #pragma omp parallel for shared(arr)
        for (int j = i % 2; j < n - 1; j += 2) {

            if (arr[j] > arr[j + 1]) {

                // Swap adjacent elements
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}


/*========================================================
                    MERGE FUNCTION
==========================================================*/

void merge(vector<int>& arr, int left, int mid, int right) {

    int n1 = mid - left + 1;
    int n2 = right - mid;

    vector<int> leftArr(n1);
    vector<int> rightArr(n2);

    // Copy left half
    for (int i = 0; i < n1; i++) {
        leftArr[i] = arr[left + i];
    }

    // Copy right half
    for (int j = 0; j < n2; j++) {
        rightArr[j] = arr[mid + 1 + j];
    }

    int i = 0;
    int j = 0;
    int k = left;

    // Merge sorted arrays
    while (i < n1 && j < n2) {

        if (leftArr[i] <= rightArr[j]) {
            arr[k++] = leftArr[i++];
        }
        else {
            arr[k++] = rightArr[j++];
        }
    }

    // Copy remaining elements
    while (i < n1) {
        arr[k++] = leftArr[i++];
    }

    while (j < n2) {
        arr[k++] = rightArr[j++];
    }
}


/*========================================================
                PARALLEL MERGE SORT
==========================================================*/

/*
Optimized Merge Sort:
- Uses OpenMP Tasks
- Better than parallel sections for recursion
- Avoids excessive thread creation
*/

void parallelMergeSort(vector<int>& arr, int left, int right) {

    // Base condition
    if (left >= right)
        return;

    // Safer midpoint calculation
    int mid = left + (right - left) / 2;

    /*
    Threshold:
    Small arrays are processed sequentially
    to reduce parallel overhead
    */

    if (right - left < 1000) {

        parallelMergeSort(arr, left, mid);
        parallelMergeSort(arr, mid + 1, right);
    }
    else {

        #pragma omp task shared(arr)
        {
            parallelMergeSort(arr, left, mid);
        }

        #pragma omp task shared(arr)
        {
            parallelMergeSort(arr, mid + 1, right);
        }

        // Wait for tasks to complete
        #pragma omp taskwait
    }

    // Merge sorted halves
    merge(arr, left, mid, right);
}


/*========================================================
                    PRINT FUNCTION
==========================================================*/

void printArray(const vector<int>& arr) {

    for (int x : arr) {
        cout << x << " ";
    }

    cout << endl;
}


/*========================================================
                    MAIN FUNCTION
==========================================================*/

int main() {

    // Set number of threads
    omp_set_num_threads(4);

    // Original array
    vector<int> arr1 = {64, 34, 25, 12, 22, 11, 90};
    vector<int> arr2 = arr1;

    /*====================================================
                PARALLEL BUBBLE SORT
    =====================================================*/

    cout << "Original Array for Bubble Sort:\n";
    printArray(arr1);

    double startBubble = omp_get_wtime();

    parallelBubbleSort(arr1);

    double endBubble = omp_get_wtime();

    cout << "\nSorted Array using Parallel Bubble Sort:\n";
    printArray(arr1);

    cout << "Bubble Sort Execution Time: "
         << (endBubble - startBubble)
         << " seconds\n";


    /*====================================================
                PARALLEL MERGE SORT
    =====================================================*/

    cout << "\nOriginal Array for Merge Sort:\n";
    printArray(arr2);

    double startMerge = omp_get_wtime();

    /*
    Parallel region required for OpenMP tasks
    */

    #pragma omp parallel
    {
        #pragma omp single
        {
            parallelMergeSort(arr2, 0, arr2.size() - 1);
        }
    }

    double endMerge = omp_get_wtime();

    cout << "\nSorted Array using Parallel Merge Sort:\n";
    printArray(arr2);

    cout << "Merge Sort Execution Time: "
         << (endMerge - startMerge)
         << " seconds\n";


    /*====================================================
                    COMPARISON
    =====================================================*/

    cout << "\n========== DIFFERENCE ==========\n";

    cout << "1. Bubble Sort Complexity  : O(n^2)\n";
    cout << "2. Merge Sort Complexity   : O(n log n)\n";

    cout << "3. Bubble Sort is slower for large datasets\n";
    cout << "4. Merge Sort is faster and efficient\n";

    cout << "5. Bubble Sort uses adjacent swapping\n";
    cout << "6. Merge Sort uses divide and conquer\n";

    cout << "7. Merge Sort is better suited for parallelism\n";

    return 0;
}
