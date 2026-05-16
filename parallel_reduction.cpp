#include <iostream>
#include <vector>
#include <omp.h>

using namespace std;

// Parallel Reduction for Minimum
int parallelMin(vector<int>& arr) {

    int minVal = arr[0];

    #pragma omp parallel for reduction(min:minVal)
    for (int i = 0; i < (int)arr.size(); i++) {

        if (arr[i] < minVal)
            minVal = arr[i];
    }

    return minVal;
}

// Parallel Reduction for Maximum
int parallelMax(vector<int>& arr) {

    int maxVal = arr[0];

    #pragma omp parallel for reduction(max:maxVal)
    for (int i = 0; i < (int)arr.size(); i++) {

        if (arr[i] > maxVal)
            maxVal = arr[i];
    }

    return maxVal;
}

// Parallel Reduction for Sum
int parallelSum(vector<int>& arr) {

    int sum = 0;

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < (int)arr.size(); i++) {

        sum += arr[i];
    }

    return sum;
}

// Parallel Reduction for Average
double parallelAverage(vector<int>& arr) {

    int sum = parallelSum(arr);

    return (double)sum / arr.size();
}

int main() {

    int n;

    cout << "Enter number of elements: ";
    cin >> n;

    // Input validation
    if (n <= 0) {

        cout << "Invalid array size!" << endl;
        return 0;
    }

    vector<int> arr(n);

    cout << "Enter elements:\n";

    for (int i = 0; i < n; i++) {
        cin >> arr[i];
    }

    cout << "\nThreads Used: "
         << omp_get_max_threads() << endl;

    int minVal = parallelMin(arr);
    int maxVal = parallelMax(arr);
    int sumVal = parallelSum(arr);
    double avgVal = parallelAverage(arr);

    cout << "\nResults using Parallel Reduction:\n";

    cout << "Minimum = " << minVal << endl;
    cout << "Maximum = " << maxVal << endl;
    cout << "Sum     = " << sumVal << endl;
    cout << "Average = " << avgVal << endl;

    return 0;
}
