#include <iostream>
#include <array>
#include <cstdint>
#include <chrono>

using patch_array = std::array<uint8_t, 16 * 16 * 3>;
using return_array = std::array<uint8_t, 2 * 2 * 3>;

void max_pool(const patch_array &arr_patch,
              return_array &arr_buffer);

int to1D(int x, int y, int z, int patch_size);

int main()
{
    patch_array myPatch;
    return_array myBuffer;

    myPatch = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5};

    auto begin = std::chrono::steady_clock::now();
    max_pool(myPatch, myBuffer);
    auto end = std::chrono::steady_clock::now();

    std::cout << "Time = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;

    // std::cout << "MP = " << myBuffer <<std::endl;
    // std::for_each(std::begin(myBuffer), std::end(myBuffer), print);
    int i;
    for (i = 0; i < 2 * 2 * 3; i++)
        std::cout << static_cast<int>(myBuffer[i]) << ' ';

    return 0;
}

void max_pool(const patch_array &arr_patch,
              return_array &arr_buffer)
{
    int nr_channels, out_size, patch_size, pool_size, mp_i, mp_j, ch, p_i, p_j, index;
    uint8_t maximum;

    nr_channels = 3;
    out_size = 2;
    patch_size = 16;
    pool_size = 8;

    for (mp_i = 0; mp_i < out_size; mp_i++)
    {
        for (mp_j = 0; mp_j < out_size; mp_j++)
        {
            for (ch = 0; ch < nr_channels; ch++)
            {
                maximum = 0;
                for (p_i = 0; p_i < pool_size; p_i++)
                {
                    for (p_j = 0; p_j < pool_size; p_j++)
                    {
                        index = to1D(p_i + pool_size * mp_i, p_j + pool_size * mp_j, ch, patch_size);
                        if (arr_patch[index] > maximum)
                        {
                            maximum = arr_patch[index];
                        }
                    }
                    arr_buffer[to1D(mp_i, mp_j, ch, out_size)] = maximum;
                }
            }
        }
    }
}

int to1D(int x, int y, int z, int patch_size)
{
    return (z * patch_size * patch_size) + (y * patch_size) + x;
}