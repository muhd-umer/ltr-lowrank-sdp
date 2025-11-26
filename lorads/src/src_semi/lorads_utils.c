#include "lorads_utils.h"

#include <math.h>
#ifdef __WIN32
#include <Windows.h>
#else
#include <sys/time.h>
#include <sys/stat.h>
#endif
#include <signal.h>
#include <errno.h>
#include <limits.h>
#include <string.h>

#ifdef MEMDEBUG
#include "memwatch.h"
#endif

#ifdef __WIN32
static BOOL monitorCtrlC(DWORD fdwCtrlType)
{
    switch (fdwCtrlType)
    {
    case CTRL_C_EVENT:
        exit(0);
        return TRUE;
        break;
    default:
        return FALSE;
        break;
    }
}
#else
static int isCtrlC = 0;
static void monitorCtrlC(int sigNum)
{
    isCtrlC = 1;
    exit(0);
    return;
}
static struct sigaction act;
#endif
/* TODO: Add compatibility for Windows platform */
#ifdef __WIN32
static double my_clock(void)
{
    LARGE_INTEGER frequency;
    LARGE_INTEGER currentTime;

    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&currentTime);

    return (double)currentTime.QuadPart / (double)frequency.QuadPart;
}
#else
static double my_clock(void)
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return (1e-06 * t.tv_usec + t.tv_sec);
}
#endif

static lorads_int mkdir_recursive(const char *path)
{
    struct stat st;
    if (stat(path, &st) == 0)
    {
        return S_ISDIR(st.st_mode) ? 0 : -1;
    }
    if (errno != ENOENT)
    {
        return -1;
    }
    char *parent = strdup(path);
    if (!parent)
    {
        return -1;
    }
    char *slash = strrchr(parent, '/');
    if (slash && slash != parent)
    {
        *slash = '\0';
        if (strlen(parent) > 0 && mkdir_recursive(parent) != 0)
        {
            free(parent);
            return -1;
        }
    }
    free(parent);
    return mkdir(path, 0755);
}

extern lorads_int LUtilEnsureDir(const char *path)
{
    if (!path)
    {
        return -1;
    }
    return mkdir_recursive(path);
}

/**
 * @brief Partition function for integer indices and double values
 * @param ind Array of integer indices
 * @param val Array of double values
 * @param l Left boundary index
 * @param h Right boundary index
 * @return Partition index
 * @details Implements partitioning for quicksort with integer indices and double values:
 * - Uses first element as pivot
 * - Maintains index-value pairs during partitioning
 * - Returns final position of pivot
 */
static lorads_int dpartitioni(lorads_int *ind, double *val, lorads_int l, lorads_int h)
{

    double tmp2 = 0.0, p = val[l];
    lorads_int tmp = l, tmp3 = 0;

    while (l < h)
    {

        while (l < h && val[h] >= p)
        {
            --h;
        }
        while (l < h && val[l] <= p)
        {
            ++l;
        }

        if (l < h)
        {
            tmp2 = val[l];
            val[l] = val[h];
            val[h] = tmp2;
            tmp3 = ind[l];
            ind[l] = ind[h];
            ind[h] = tmp3;
        }
    }

    tmp2 = val[l];
    val[l] = val[tmp];
    val[tmp] = tmp2;
    tmp3 = ind[l];
    ind[l] = ind[tmp];
    ind[tmp] = tmp3;

    return l;
}

/**
 * @brief Partition function for double indices and integer values
 * @param ind Array of double indices
 * @param val Array of integer values
 * @param l Left boundary index
 * @param h Right boundary index
 * @return Partition index
 * @details Implements partitioning for quicksort with double indices and integer values:
 * - Uses first element as pivot
 * - Maintains index-value pairs during partitioning
 * - Returns final position of pivot
 */
static lorads_int ipartitiond(double *ind, lorads_int *val, lorads_int l, lorads_int h)
{

    lorads_int tmp2 = 0, p = val[l], tmp = l;
    double tmp3;

    while (l < h)
    {
        while (l < h && val[h] >= p)
        {
            --h;
        }
        while (l < h && val[l] <= p)
        {
            ++l;
        }

        if (l < h)
        {
            tmp2 = val[l];
            val[l] = val[h];
            val[h] = tmp2;
            tmp3 = ind[l];
            ind[l] = ind[h];
            ind[h] = tmp3;
        }
    }

    tmp2 = val[l];
    val[l] = val[tmp];
    val[tmp] = tmp2;
    tmp3 = ind[l];
    ind[l] = ind[tmp];
    ind[tmp] = tmp3;

    return l;
}

/**
 * @brief Partition function for integer indices and integer values
 * @param ind Array of integer indices
 * @param val Array of integer values
 * @param l Left boundary index
 * @param h Right boundary index
 * @return Partition index
 * @details Implements partitioning for quicksort with integer indices and values:
 * - Uses first element as pivot
 * - Maintains index-value pairs during partitioning
 * - Returns final position of pivot
 */
static lorads_int ipartitioni(lorads_int *ind, lorads_int *val, lorads_int l, lorads_int h)
{

    lorads_int tmp = l, tmp2 = 0, tmp3, p = val[l];

    while (l < h)
    {

        while (l < h && val[h] <= p)
        {
            --h;
        }
        while (l < h && val[l] >= p)
        {
            ++l;
        }

        if (l < h)
        {
            tmp2 = val[l];
            val[l] = val[h];
            val[h] = tmp2;
            tmp3 = ind[l];
            ind[l] = ind[h];
            ind[h] = tmp3;
        }
    }

    tmp2 = val[l];
    val[l] = val[tmp];
    val[tmp] = tmp2;
    tmp3 = ind[l];
    ind[l] = ind[tmp];
    ind[tmp] = tmp3;

    return l;
}

/**
 * @brief Partition function for double indices and double values
 * @param ind Array of double indices
 * @param val Array of double values
 * @param l Left boundary index
 * @param h Right boundary index
 * @return Partition index
 * @details Implements partitioning for quicksort with double indices and values:
 * - Uses first element as pivot
 * - Maintains index-value pairs during partitioning
 * - Returns final position of pivot
 */
static lorads_int dpartitiond(double *ind, double *val, lorads_int l, lorads_int h)
{

    lorads_int tmp = l;
    double tmp2 = 0.0, tmp3, p = val[l];

    while (l < h)
    {

        while (l < h && val[h] >= p)
        {
            --h;
        }
        while (l < h && val[l] <= p)
        {
            ++l;
        }

        if (l < h)
        {
            tmp2 = val[l];
            val[l] = val[h];
            val[h] = tmp2;
            tmp3 = ind[l];
            ind[l] = ind[h];
            ind[h] = tmp3;
        }
    }

    tmp2 = val[l];
    val[l] = val[tmp];
    val[tmp] = tmp2;
    tmp3 = ind[l];
    ind[l] = ind[tmp];
    ind[tmp] = tmp3;

    return l;
}

/**
 * @brief Get the current timestamp in seconds.
 * @return Current time as a double (seconds).
 */
extern double LUtilGetTimeStamp(void)
{

    return my_clock();
}

/**
 * @brief Symmetrize a matrix by copying lower triangular elements to upper triangular
 * @param n Matrix dimension
 * @param v Matrix data stored in column-major format
 * @details Makes a matrix symmetric by copying lower triangular elements to upper triangular:
 * - Assumes lower triangular is filled
 * - Copies elements to maintain symmetry
 * - Matrix is stored in column-major format
 */
extern void LUtilMatSymmetrize(lorads_int n, double *v)
{

    for (lorads_int i = 0, j; i < n; ++i)
    {
        for (j = i + 1; j < n; ++j)
        {
            FULL_ENTRY(v, n, i, j) = FULL_ENTRY(v, n, j, i);
        }
    }

    return;
}

/**
 * @brief Print double array contents
 * @param n Array length
 * @param d Array to print
 * @details Prints array elements in scientific notation with 3 decimal places
 */
extern void LUtilPrintDblContent(lorads_int n, double *d)
{

    for (lorads_int i = 0; i < n; ++i)
    {
        printf("%5.3e, ", d[i]);
    }
    printf("\n");
    return;
}

/**
 * @brief Calculate sum of double array elements
 * @param n Array length
 * @param d Input array
 * @return Sum of array elements
 * @details Computes sum of all elements in the array
 */
extern double LUtilPrintDblSum(lorads_int n, double *d)
{

    double ds = 0.0;

    for (lorads_int i = 0; i < n; ++i)
    {
        ds += d[i];
    }

    return ds;
}

/**
 * @brief Calculate sum of absolute values in double array
 * @param n Array length
 * @param d Input array
 * @return Sum of absolute values
 * @details Computes sum of absolute values of all elements in the array
 */
extern double LUtilPrintDblAbsSum(lorads_int n, double *d)
{

    double ds = 0.0;

    for (lorads_int i = 0; i < n; ++i)
    {
        ds += fabs(d[i]);
    }

    return ds;
}

/* Sorting */
/**
 * @brief Check if an integer array is sorted in ascending order.
 * @param n Length of the array.
 * @param idx Array to check.
 * @return 1 if sorted, 0 otherwise.
 */
extern lorads_int LUtilCheckIfAscending(lorads_int n, lorads_int *idx)
{
    /* Check is an lorads_integer array is ascending. */

    for (lorads_int i = 0; i < n - 1; ++i)
    {
        if (idx[i] > idx[i + 1])
        {
            return 0;
        }
    }

    return 1;
}

// extern void LUtilSortIntbyDbl( lorads_int *data, double *ref, lorads_int low, lorads_int up ) {

//     if ( low < up ) {
//         lorads_int p = dpartitioni(data, ref, low, up);
//         LUtilSortIntbyDbl(data, ref, low, p - 1);
//         LUtilSortIntbyDbl(data, ref, p + 1, up);
//     }

//     return;
// }

/**
 * @brief Sort an integer array in descending order based on a reference array.
 * @param data Array to sort.
 * @param ref Reference array for sorting.
 * @param low Start index.
 * @param up End index.
 */
extern void LUtilDescendSortIntByInt(lorads_int *data, lorads_int *ref, lorads_int low, lorads_int up)
{

    if (low < up)
    {
        lorads_int p = ipartitioni(data, ref, low, up);
        LUtilDescendSortIntByInt(data, ref, low, p - 1);
        LUtilDescendSortIntByInt(data, ref, p + 1, up);
    }

    return;
}

/**
 * @brief Sort an integer array based on a double reference array.
 * @param data Array to sort.
 * @param ref Reference array for sorting.
 * @param low Start index.
 * @param up End index.
 */
extern void LUtilAscendSortDblByInt(double *data, lorads_int *ref, lorads_int low, lorads_int up)
{

    if (low < up)
    {
        lorads_int p = ipartitiond(data, ref, low, up);
        LUtilAscendSortDblByInt(data, ref, low, p - 1);
        LUtilAscendSortDblByInt(data, ref, p + 1, up);
    }

    return;
}

/**
 * @brief Sort a double array in ascending order based on an integer reference array.
 * @param data Array to sort.
 * @param ref Reference array for sorting.
 * @param low Start index.
 * @param up End index.
 */
extern void LUtilSortDblByDbl(double *data, double *ref, lorads_int low, lorads_int up)
{

    if (low < up)
    {
        lorads_int p = dpartitiond(data, ref, low, up);
        LUtilSortDblByDbl(data, ref, low, p - 1);
        LUtilSortDblByDbl(data, ref, p + 1, up);
    }

    return;
}
#ifdef __WIN32
extern void LUtilStartCtrlCCheck(void)
{
    SetConsoleCtrlHandler((PHANDLER_ROUTINE)monitorCtrlC, TRUE);
}
#else
extern void LUtilStartCtrlCCheck(void)
{

    act.sa_handler = monitorCtrlC;
    sigaction(SIGINT, &act, NULL);

    return;
}
extern lorads_int LUtilCheckCtrlC(void)
{

    return isCtrlC;
}

extern void LUtilResetCtrl(void)
{

    isCtrlC = 0;
}
#endif

// extern lorads_int MKL_Get_Max_Threads( void );
// extern lorads_int MKL_Set_Num_Threads( lorads_int nth );

/**
 * @brief Get the number of global MKL threads.
 * @return Number of threads.
 */
//
// extern lorads_int LUtilGetGlobalMKLThreads( void ) {
//
//     return MKL_Get_Max_Threads();
// }
//
// extern void LUtilSetGlobalMKLThreads( lorads_int nTargetThreads ) {
//
//     MKL_Set_Num_Threads(nTargetThreads);
//
//     return;
// }

/**
 * @brief Initialize array with values from 1/n to 1
 * @param var Array to initialize
 * @param size Array size
 * @details Initializes array with values i/size for i from 1 to size
 */
extern void LORADS_ONE(double *var, lorads_int size)
{
    for (lorads_int i = 0; i < size; ++i)
    {
        var[i] = (double)(i + 1) / (double)size;
        //        if (i == 0){
        //            var[i] = 1;
        //        }else{
        //            var[i] = 0;
        //        }
    }
}

/**
 * @brief Update an exponential moving average (EMA) with a new value.
 * @param current_ema Pointer to the current EMA value.
 * @param old_ema Pointer to the previous EMA value.
 * @param new_value New value to incorporate.
 * @param alpha Smoothing factor.
 * @param threshold Threshold for update.
 * @param update_interval Update interval.
 * @param counter Pointer to update counter.
 * @return 1 if updated, 0 otherwise.
 */
extern lorads_int LUtilUpdateCheckEma(double *current_ema, double *old_ema, double new_value, double alpha, double threshold, lorads_int update_interval, lorads_int *counter)
{
    lorads_int result = 1;                                                  // Default to true
    *current_ema = alpha * new_value + (1 - alpha) * (*current_ema); // Update the EMA value

    // Check if it's time to update the old EMA
    if (*counter >= update_interval)
    {
        // printf("current_ema: %.8f\n", *current_ema);
        // printf("old_ema: %.8f\n", *old_ema);
        // printf("counter: %d\n", *counter);
        // Avoid division by zero if old_ema is 0 (e.g., the first update)
        if (*old_ema != 0)
        {
            double change = (*current_ema - *old_ema) / *old_ema;     // Calculate the proportional change
            // printf("change: %.8f\n", change);
            result = (change >= -threshold) && (change <= threshold); // Determine if change is within threshold bounds
            // printf("ratio: %.4f\n", result);
        }
        *old_ema = *current_ema; // Update the old EMA value
        // printf("old_ema: %.8f\n", *old_ema);

        *counter = 1;            // Reset the counter
    }
    else
    {
        (*counter)++;
    }

    return result;
}


// extern int AUtilUpdateCheckEma(double *current_ema, double *old_ema, double new_value, double alpha, double threshold, int update_interval, int *counter)
// {
//     int result = 1;                                                  // Default to true
//     *current_ema = alpha * new_value + (1 - alpha) * (*current_ema); // Update the EMA value

//     // Check if it's time to update the old EMA
//     if (*counter >= update_interval)
//     {
//         // Avoid division by zero if old_ema is 0 (e.g., the first update)
//         if (*old_ema != 0)
//         {
//             double change = (*current_ema - *old_ema) / *old_ema;     // Calculate the proportional change
//             result = (change >= -threshold) && (change <= threshold); // Determine if change is within threshold bounds
//             // printf("ratio: %.4f\n", result);
//         }
//         *old_ema = *current_ema; // Update the old EMA value
//         *counter = 1;            // Reset the counter
//     }
//     else
//     {
//         (*counter)++;
//     }

//     return result;
// }

/**
 * @brief Reallocate a double array to a new size
 * @param data Pointer to array pointer
 * @param nOld Old size
 * @param nNew New size
 * @details Reallocates array while preserving existing data:
 * - Allocates new array of size nNew
 * - Copies nOld elements from old array
 * - Frees old array
 * - Updates array pointer
 */
extern void REALLOC(double **data, lorads_int nOld, lorads_int nNew){
    double *dataNewPtr;
    LORADS_INIT(dataNewPtr, double, nNew);
    LORADS_MEMCPY(dataNewPtr, *data, double, nOld);
    LORADS_FREE(*data);
    *data = dataNewPtr;
}
