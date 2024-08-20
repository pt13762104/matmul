#include <chrono>
#include <iostream>
#include <omp.h>
#include <random>
#define type float
typedef type vec __attribute__((vector_size(32)));
using namespace std;
using namespace chrono;
high_resolution_clock Clock;
mt19937_64 rng(steady_clock::now().time_since_epoch().count());
const int SZ = 1920;
const int B = sizeof(vec) / sizeof(type);
int main()
{
    type a[SZ][SZ];
    vec b[SZ][SZ / B], c[SZ][SZ / B];
    type bb[SZ][SZ], cc[SZ][SZ];
    auto t0 = Clock.now();
    for (int i = 0; i < SZ; i++)
        for (int j = 0; j < SZ / B; j++)
        {
            b[i][j] = vec{
                uniform_real_distribution<type>(0, SZ)(rng), uniform_real_distribution<type>(0, SZ)(rng),
                uniform_real_distribution<type>(0, SZ)(rng), uniform_real_distribution<type>(0, SZ)(rng),
                uniform_real_distribution<type>(0, SZ)(rng), uniform_real_distribution<type>(0, SZ)(rng),
                uniform_real_distribution<type>(0, SZ)(rng), uniform_real_distribution<type>(0, SZ)(rng),
            };
            c[i][j] = vec{0, 0, 0, 0, 0, 0, 0, 0};
            for (int k = 0; k < B; k++)
            {
                a[i][j * B + k] = uniform_real_distribution<type>(0, SZ)(rng);
                bb[i][j * B + k] = b[i][j][k];
                cc[i][j * B + k] = 0;
            }
        }
    auto t1 = Clock.now();
    cout << "Init time: " << duration_cast<milliseconds>(t1 - t0).count() << " ms" << endl;
    double sum = 0;
    auto tn0 = Clock.now();
    for (int i = 0; i < SZ; i++)
        for (int j = 0; j < SZ; j++)
            for (int k = 0; k < SZ; k++)
                cc[i][j] += a[i][k] * bb[k][j];
    auto tn1 = Clock.now();
    for (int i = 0; i < SZ; i++)
        for (int j = 0; j < SZ; j++)
        {
            sum += cc[i][j];
            cc[i][j] = 0;
        }
    cout << sum << endl;
    cout << "Naive time: " << duration_cast<milliseconds>(tn1 - tn0).count() << " ms" << endl;
    cout << "Naive GFLOPS: " << ((double)SZ * SZ * SZ) / duration_cast<nanoseconds>(tn1 - tn0).count() << endl;
    sum = 0;
    auto t2 = Clock.now();
    for (int i = 0; i < SZ; i++)
        for (int k = 0; k < SZ; k++)
            for (int j = 0; j < SZ; j++)
                cc[i][j] += a[i][k] * bb[k][j];
    auto t3 = Clock.now();
    for (int i = 0; i < SZ; i++)
        for (int j = 0; j < SZ; j++)
            sum += cc[i][j];
    cout << sum << endl;
    cout << "Naive (reordered, AVX) time: " << duration_cast<milliseconds>(t3 - t2).count() << " ms" << endl;
    cout << "Naive (reordered, AVX) GFLOPS: " << ((double)SZ * SZ * SZ) / duration_cast<nanoseconds>(t3 - t2).count()
         << endl;
    cout << "Speedup: "
         << (double)duration_cast<nanoseconds>(tn1 - tn0).count() / duration_cast<nanoseconds>(t3 - t2).count() << "x"
         << endl;
    sum = 0;
    auto t4 = Clock.now();
    for (int i3 = 0; i3 < SZ / B; i3 += 8)
        for (int i2 = 0; i2 < SZ; i2 += 32)
            for (int h = 0; h < SZ; h += 240)
                for (int i = i2; i < i2 + 32; i += 4)
                    for (int j = i3; j < i3 + 8; j += 2)
                    {
                        vec t00 = c[i + 0][j + 0];
                        vec t01 = c[i + 0][j + 1];
                        vec t10 = c[i + 1][j + 0];
                        vec t11 = c[i + 1][j + 1];
                        vec t20 = c[i + 2][j + 0];
                        vec t21 = c[i + 2][j + 1];
                        vec t30 = c[i + 3][j + 0];
                        vec t31 = c[i + 3][j + 1];
                        for (int k = h; k < h + 240; k++)
                        {
                            vec a0 = vec{} + a[i + 0][k];
                            vec a1 = vec{} + a[i + 1][k];
                            vec a2 = vec{} + a[i + 2][k];
                            vec a3 = vec{} + a[i + 3][k];
                            t00 += a0 * b[k][j + 0];
                            t01 += a0 * b[k][j + 1];
                            t10 += a1 * b[k][j + 0];
                            t11 += a1 * b[k][j + 1];
                            t20 += a2 * b[k][j + 0];
                            t21 += a2 * b[k][j + 1];
                            t30 += a3 * b[k][j + 0];
                            t31 += a3 * b[k][j + 1];
                        }
                        c[i + 0][j + 0] = t00;
                        c[i + 0][j + 1] = t01;
                        c[i + 1][j + 0] = t10;
                        c[i + 1][j + 1] = t11;
                        c[i + 2][j + 0] = t20;
                        c[i + 2][j + 1] = t21;
                        c[i + 3][j + 0] = t30;
                        c[i + 3][j + 1] = t31;
                    }
    auto t5 = Clock.now();
    for (int i = 0; i < SZ; i++)
        for (int j = 0; j < SZ / B; j++)
            for (int k = 0; k < B; k++)
                sum += c[i][j][k];
    cout << sum << endl;
    cout << "Naive (reordered, AVX, kernel) time: " << duration_cast<milliseconds>(t5 - t4).count() << " ms" << endl;
    cout << "Naive (reordered, AVX, kernel) GFLOPS: "
         << ((double)SZ * SZ * SZ) / duration_cast<nanoseconds>(t5 - t4).count() << endl;
    cout << "Speedup: "
         << (double)duration_cast<nanoseconds>(tn1 - tn0).count() / duration_cast<nanoseconds>(t5 - t4).count() << "x"
         << endl;
}
