%%writefile salsa20_128.cu
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#define SALSA_ROUNDS 10              // 10 double rounds = Salsa20/20
#define BYTES_PER_BLOCK 64

// ================= Utility functions =================

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

// ================= Salsa20 core =================

__device__ void salsa20_core(const uint32_t in[16], uint8_t out[64])
{
    uint32_t x[16];

    #pragma unroll
    for (int i = 0; i < 16; i++) x[i] = in[i];

    for (int i = 0; i < SALSA_ROUNDS; i++) {
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

    #pragma unroll
    for (int i = 0; i < 16; i++) {
        U32TO8_LE(out + 4 * i, x[i] + in[i]);
    }
}

// ================= Salsa20-128 kernel =================

__global__ void salsa20_128_xor_kernel(const uint32_t *key, const uint32_t *nonce, uint64_t base_counter, uint8_t *buf, size_t data_len)
{
    const uint8_t sigma[17] = "expand 16-byte k";

    uint64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t offset = tid * BYTES_PER_BLOCK;
    if (offset >= data_len) return;

    uint32_t state[16];

    // constants
    state[0]  = 0x61707865;
    state[5]  = 0x3120646e;
    state[10] = 0x79622d36;
    state[15] = 0x6b206574;

    // 128-bit key repeated
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        state[1 + i]  = key[i];
        state[11 + i] = key[i];
    }

    // counter
    uint64_t ctr = base_counter + tid;
    state[6] = (uint32_t)ctr;
    state[7] = (uint32_t)(ctr >> 32);

    // nonce
    state[8] = nonce[0];
    state[9] = nonce[1];

    uint8_t keystream[64];
    salsa20_core(state, keystream);

    size_t max_i = BYTES_PER_BLOCK;
    if (offset + max_i > data_len)
        max_i = data_len - offset;

    #pragma unroll
    for (size_t i = 0; i < max_i; i++) {
        buf[offset + i] ^= keystream[i];
    }
}

// ================= CUDA error check =================

void checkCuda(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// ================= Main =================

int main()
{
    int w, h, c;
    unsigned char *img = stbi_load("new.png", &w, &h, &c, 0);
    if (!img) {
        printf("Image load failed\n");
        return 1;
    }

    size_t data_len = w * h * c;
    printf("Image: %dx%d  channels=%d\n", w, h, c);

    // 128-bit key
    uint32_t key[4] = {
        0x03020100, 0x07060504,
        0x0b0a0908, 0x0f0e0d0c
    };

    uint32_t nonce[2] = {0x00000000, 0x4a000000};
    uint64_t base_counter = 0;

    uint32_t *d_key, *d_nonce;
    uint8_t *d_buf;

    checkCuda(cudaMalloc(&d_key, sizeof(key)), "malloc key");
    checkCuda(cudaMalloc(&d_nonce, sizeof(nonce)), "malloc nonce");
    checkCuda(cudaMalloc(&d_buf, data_len), "malloc buffer");

    checkCuda(cudaMemcpy(d_key, key, sizeof(key), cudaMemcpyHostToDevice), "copy key");
    checkCuda(cudaMemcpy(d_nonce, nonce, sizeof(nonce), cudaMemcpyHostToDevice), "copy nonce");
    checkCuda(cudaMemcpy(d_buf, img, data_len, cudaMemcpyHostToDevice), "copy image");

//    int threads = 64;
//    int blocks = data_len/1048576;
//    if (blocks<=0) blocks++;

    const int threads_per_block = 1024;
    size_t total_salsa_blocks =(data_len + BYTES_PER_BLOCK - 1) / BYTES_PER_BLOCK;
    int num_blocks =(total_salsa_blocks + threads_per_block - 1) / threads_per_block;

    printf("Total blocks: %d\n", total_salsa_blocks);
    printf("Number of blocks: %d\n", num_blocks);
    printf("Number of threads per block: %d\n", threads_per_block);

    cudaEvent_t start1, stop1, start2, stop2;
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);

    //Encryption
    cudaEventRecord(start1);
    salsa20_128_xor_kernel<<<num_blocks, threads_per_block>>>(d_key, d_nonce, base_counter, d_buf, data_len);
    cudaEventRecord(stop1);
    cudaEventSynchronize(stop1);

    float ms1;
    cudaEventElapsedTime(&ms1, start1, stop1);
    printf("Encryption time: %.3f ms\n", ms1);

    unsigned char *cipher = (unsigned char*)malloc(data_len);
    checkCuda(cudaMemcpy(cipher, d_buf, data_len, cudaMemcpyDeviceToHost), "copy back cipher");
    stbi_write_png("new_encrypted.png", w, h, c, cipher, w * c);

    // Decryption
    cudaEventCreate(&start2);
    cudaEventCreate(&stop2);
    cudaEventRecord(start2);
    salsa20_128_xor_kernel<<<num_blocks, threads_per_block>>>(d_key, d_nonce, base_counter, d_buf, data_len);
    cudaEventRecord(stop2);
    cudaEventSynchronize(stop2);

    float ms2;
    cudaEventElapsedTime(&ms2, start2, stop2);
    printf("Decryption time: %.3f ms\n", ms2);
    checkCuda(cudaMemcpy(img, d_buf, data_len, cudaMemcpyDeviceToHost), "copy back plain");
    stbi_write_png("new_decrypted.png", w, h, c, img, w * c);

    printf("Decryption successful\n");

    cudaFree(d_key);
    cudaFree(d_nonce);
    cudaFree(d_buf);
    free(cipher);
    stbi_image_free(img);
    cudaEventDestroy(start1);
    cudaEventDestroy(stop1);
    cudaEventDestroy(start2);
    cudaEventDestroy(stop2);
    return 0;
}
