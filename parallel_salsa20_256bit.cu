%%writefile salsa20_256.cu
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#define SALSA_ROUNDS 10
#define BYTES_PER_BLOCK 64

__device__ __host__ inline uint32_t rotl(uint32_t x, int n)
{
    return (x << n) | (x >> (32 - n));
}

__device__ __host__ inline void U32TO8_LE(uint8_t *p, uint32_t v)
{
    p[0] = v & 0xff;
    p[1] = (v >> 8) & 0xff;
    p[2] = (v >> 16) & 0xff;
    p[3] = (v >> 24) & 0xff;
}

__device__ void salsa20_core(const uint32_t in[16], uint8_t out[64])
{
    uint32_t x[16];
    #pragma unroll
    for (int i = 0; i < 16; i++) x[i] = in[i];

    for (int i = 0; i < SALSA_ROUNDS; i++)
    {
        // column round
        x[4] ^= rotl(x[0] + x[12], 7);
        x[8] ^= rotl(x[4] + x[0], 9);
        x[12] ^= rotl(x[8] + x[4], 13);
        x[0] ^= rotl(x[12] + x[8], 18);

        x[9] ^= rotl(x[5] + x[1], 7);
        x[13] ^= rotl(x[9] + x[5], 9);
        x[1] ^= rotl(x[13] + x[9], 13);
        x[5] ^= rotl(x[1] + x[13], 18);

        x[14] ^= rotl(x[10] + x[6], 7);
        x[2] ^= rotl(x[14] + x[10], 9);
        x[6] ^= rotl(x[2] + x[14], 13);
        x[10] ^= rotl(x[6] + x[2], 18);

        x[3] ^= rotl(x[15] + x[11], 7);
        x[7] ^= rotl(x[3] + x[15], 9);
        x[11] ^= rotl(x[7] + x[3], 13);
        x[15] ^= rotl(x[11] + x[7], 18);

        // row round
        x[1] ^= rotl(x[0] + x[3], 7);
        x[2] ^= rotl(x[1] + x[0], 9);
        x[3] ^= rotl(x[2] + x[1], 13);
        x[0] ^= rotl(x[3] + x[2], 18);

        x[6] ^= rotl(x[5] + x[4], 7);
        x[7] ^= rotl(x[6] + x[5], 9);
        x[4] ^= rotl(x[7] + x[6], 13);
        x[5] ^= rotl(x[4] + x[7], 18);

        x[11] ^= rotl(x[10] + x[9], 7);
        x[8] ^= rotl(x[11] + x[10], 9);
        x[9] ^= rotl(x[8] + x[11], 13);
        x[10] ^= rotl(x[9] + x[8], 18);

        x[12] ^= rotl(x[15] + x[14], 7);
        x[13] ^= rotl(x[12] + x[15], 9);
        x[14] ^= rotl(x[13] + x[12], 13);
        x[15] ^= rotl(x[14] + x[13], 18);
    }

    for (int i = 0; i < 16; i++)
    {
        uint32_t z = x[i] + in[i];
        U32TO8_LE(out + 4 * i, z);
    }
}

__global__ void salsa20_xor_kernel(const uint32_t *key, const uint32_t *nonce,uint64_t base_counter, uint8_t *buf, size_t data_len)
{
    const uint8_t sigma[17] = "expand 32-byte k"; // This is 16 bytes + null terminator
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t block_index = (uint64_t)thread_id;

    uint64_t data_offset = block_index * BYTES_PER_BLOCK;
    if (data_offset >= data_len) return;

    uint32_t state[16];
    state[0]  = ((uint32_t)sigma[0]) | ((uint32_t)sigma[1] << 8) |
                ((uint32_t)sigma[2] << 16) | ((uint32_t)sigma[3] << 24);
    state[5]  = ((uint32_t)sigma[4]) | ((uint32_t)sigma[5] << 8) |
                ((uint32_t)sigma[6] << 16) | ((uint32_t)sigma[7] << 24);
    state[10] = ((uint32_t)sigma[8]) | ((uint32_t)sigma[9] << 8) |
                ((uint32_t)sigma[10] << 16) | ((uint32_t)sigma[11] << 24);
    state[15] = ((uint32_t)sigma[12]) | ((uint32_t)sigma[13] << 8) |
                ((uint32_t)sigma[14] << 16) | ((uint32_t)sigma[15] << 24);

    for (int i = 0; i < 4; i++) state[1 + i] = key[i];
    for (int i = 0; i < 4; i++) state[11 + i] = key[i + 4];

    uint64_t block_counter = base_counter + block_index;
    state[6] = (uint32_t)(block_counter & 0xffffffff);
    state[7] = (uint32_t)(block_counter >> 32);
    state[8] = nonce[0];
    state[9] = nonce[1];

    uint8_t keystream[64];
    salsa20_core(state, keystream);

    size_t max_i = BYTES_PER_BLOCK;
    if (data_offset + max_i > data_len)
    max_i = (size_t)(data_len - data_offset);

    for (size_t i = 0; i < max_i; ++i)
    {
        buf[data_offset + i] ^= keystream[i];
    }
}

void checkCudaError(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main()
{
    int width, height, channels;
    unsigned char *image = stbi_load("new.png", &width, &height, &channels, 0);
    if (!image)
    {
        fprintf(stderr, "Failed to load image!\n");
        return 1;
    }
    size_t data_len = width * height * channels; // Corrected: Multiply by channels
    printf("Loaded image: %dx%d (%d channels)\n", width, height, channels);

    uint32_t key[8] = {
        0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c,
        0x13121110, 0x17161514, 0x1b1a1918, 0x1f1e1d1c
    };
    uint32_t nonce[2] = { 0x00000000, 0x4a000000 };
    uint64_t base_counter = 0;

    uint32_t *d_key, *d_nonce;
    uint8_t *d_buf;
    checkCudaError(cudaMalloc(&d_key, sizeof(key)), "cudaMalloc d_key");
    checkCudaError(cudaMalloc(&d_nonce, sizeof(nonce)), "cudaMalloc d_nonce");
    checkCudaError(cudaMalloc(&d_buf, data_len), "cudaMalloc d_buf");

    checkCudaError(cudaMemcpy(d_key, key, sizeof(key), cudaMemcpyHostToDevice), "cudaMemcpy d_key");
    checkCudaError(cudaMemcpy(d_nonce, nonce, sizeof(nonce), cudaMemcpyHostToDevice), "cudaMemcpy d_nonce");
    checkCudaError(cudaMemcpy(d_buf, image, data_len, cudaMemcpyHostToDevice), "cudaMemcpy image");

//  const int threads_per_block = 10;
//  int num_blocks = data_len/1048576;
//  if (num_blocks<=0) num_blocks++;
    const int threads_per_block = 4;
    size_t total_salsa_blocks =(data_len + BYTES_PER_BLOCK - 1) / BYTES_PER_BLOCK;
    int num_blocks =(total_salsa_blocks + threads_per_block - 1) / threads_per_block;

    printf("Total blocks: %d\n", total_salsa_blocks);
    printf("Number of blocks: %d\n", num_blocks);
    printf("Number of threads per block: %d\n", threads_per_block);

    cudaEvent_t start, stop, start1, stop1;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    salsa20_xor_kernel<<<num_blocks, threads_per_block>>>(d_key, d_nonce, base_counter, d_buf, data_len);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_ms;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    unsigned char *cipher = (unsigned char*)malloc(data_len);
    checkCudaError(cudaMemcpy(cipher, d_buf, data_len, cudaMemcpyDeviceToHost), "copy back cipher");
    stbi_write_png("new_encrypted.png", width, height, channels, cipher, width * channels);

    printf("Encryption done in %.3f ms\n", elapsed_ms);

    // Decrypt again
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);
    cudaEventRecord(start1);
    salsa20_xor_kernel<<<num_blocks, threads_per_block>>>(d_key, d_nonce, base_counter, d_buf, data_len);
    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);
    float elapsed_ms1;
    cudaEventElapsedTime(&elapsed_ms1, start1, stop1);
    unsigned char *recovered = (unsigned char*)malloc(data_len);
    checkCudaError(cudaMemcpy(recovered, d_buf, data_len, cudaMemcpyDeviceToHost), "copy back recovered");
    stbi_write_png("new_decrypted.png", width, height, channels, recovered, width * channels);

    printf("Decryption done in %.3f ms\n", elapsed_ms1);

    cudaFree(d_key);
    cudaFree(d_nonce);
    cudaFree(d_buf);
    free(image);
    free(cipher);
    free(recovered);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(start1);
    cudaEventDestroy(stop1);
    return 0;
}