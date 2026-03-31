%%writefile salsa20_128.c
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Define implementation for STB libraries
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

/* --- Salsa20 Core Functions --- */
static inline uint32_t rotl32(uint32_t x, int n) {
    return (x << n) | (x >> (32 - n));
}

#define QR(a,b,c,d)        \
    b ^= rotl32(a + d, 7); \
    c ^= rotl32(b + a, 9); \
    d ^= rotl32(c + b,13); \
    a ^= rotl32(d + c,18);

static uint32_t load32(const uint8_t *p) {
    return ((uint32_t)p[0]) | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static void store32(uint8_t *p, uint32_t v) {
    p[0] = v & 0xff; p[1] = (v >> 8) & 0xff; p[2] = (v >> 16) & 0xff; p[3] = (v >> 24) & 0xff;
}

static void salsa20_block(uint32_t out[16], const uint32_t in[16]) {
    uint32_t x[16];
    for (int i = 0; i < 16; i++) x[i] = in[i];
    for (int i = 0; i < 10; i++) {
        QR(x[0], x[4], x[8],  x[12]) QR(x[5], x[9], x[13], x[1])
        QR(x[10],x[14],x[2],  x[6])  QR(x[15],x[3], x[7],  x[11])
        QR(x[0], x[1], x[2],  x[3])  QR(x[5], x[6], x[7],  x[4])
        QR(x[10],x[11],x[8],  x[9])  QR(x[15],x[12],x[13], x[14])
    }
    for (int i = 0; i < 16; i++) out[i] = x[i] + in[i];
}

void salsa20_crypt_128(uint8_t *data, size_t len, const uint8_t key[16], const uint8_t nonce[8]) {
    static const uint8_t sigma[16] = "expand 16-byte k";
    uint32_t state[16], block[16];
    uint8_t keystream[64];

    for (size_t i = 0; i < len; i += 64) {
        state[0] = load32(sigma + 0);  state[5] = load32(sigma + 4);
        state[10] = load32(sigma + 8); state[15] = load32(sigma + 12);
        for (int j = 0; j < 4; j++) {
            uint32_t k = load32(key + j * 4);
            state[1 + j] = k; state[11 + j] = k;
        }
        state[6] = load32(nonce + 0); state[7] = load32(nonce + 4);
        state[8] = (uint32_t)(i / 64); state[9] = (uint32_t)((i / 64) >> 32);

        salsa20_block(block, state);
        for (int j = 0; j < 16; j++) store32(keystream + 4 * j, block[j]);

        for (size_t j = 0; j < 64 && (i + j) < len; j++) {
            data[i + j] ^= keystream[j];
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        printf("Usage: %s <input.png> <encrypted_output.png> <decrypted_output.png>\n", argv[0]);
        return 1;
    }

    const char *input_image_path = argv[1];
    const char *encrypted_output_path = argv[2];
    const char *decrypted_output_path = argv[3];

    int width, height, channels;
    // 1. Load PNG and decode to raw pixels
    uint8_t *original_pixels = stbi_load(input_image_path, &width, &height, &channels, 0);
    if (!original_pixels) {
        printf("Error: Could not load image %s\n", input_image_path);
        return 1;
    }

    size_t pixel_data_len = (size_t)width * height * channels;
    uint8_t key[16] = "secret_pixel_key";
    uint8_t nonce[8] = "87654321";

    printf("Image Loaded: %dx%d, %d channels\n", width, height, channels);

    // 4. Allocate new buffers
    uint8_t *encrypted_pixels = (uint8_t *)malloc(pixel_data_len);
    uint8_t *decrypted_pixels = (uint8_t *)malloc(pixel_data_len);

    if (!encrypted_pixels || !decrypted_pixels) {
        printf("Error: Failed to allocate memory.\n");
        stbi_image_free(original_pixels);
        free(encrypted_pixels);
        free(decrypted_pixels);
        return 1;
    }

    // 5. Copy original_pixels to encrypted_pixels
    memcpy(encrypted_pixels, original_pixels, pixel_data_len);

    printf("Encrypting pixels...\n");
    // 6. Time the encryption process:
    clock_t start_enc = clock();
    salsa20_crypt_128(encrypted_pixels, pixel_data_len, key, nonce);
    clock_t end_enc = clock();
    double enc_time_sec = (double)(end_enc - start_enc) / CLOCKS_PER_SEC;
    printf("Encryption time: %f sec\n", enc_time_sec);

    // 7. Save the encrypted pixels
    if (stbi_write_png(encrypted_output_path, width, height, channels, encrypted_pixels, width * channels)) {
        printf("Success! Encrypted PNG saved as %s\n", encrypted_output_path);
    } else {
        printf("Error: Could not save encrypted image %s.\n", encrypted_output_path);
    }

    // 8. Copy encrypted_pixels to decrypted_pixels
    memcpy(decrypted_pixels, encrypted_pixels, pixel_data_len);

    printf("Decrypting pixels...\n");
    // 9. Time the decryption process:
    clock_t start_dec = clock();
    salsa20_crypt_128(decrypted_pixels, pixel_data_len, key, nonce); // Same function decrypts
    clock_t end_dec = clock();
    double dec_time_sec = (double)(end_dec - start_dec) / CLOCKS_PER_SEC;
    printf("Decryption time: %f sec\n", dec_time_sec);

    // 10. Save the decrypted pixels
    if (stbi_write_png(decrypted_output_path, width, height, channels, decrypted_pixels, width * channels)) {
        printf("Success! Decrypted PNG saved as %s\n", decrypted_output_path);
    }
    else {
        printf("Error: Could not save decrypted image %s.\n", decrypted_output_path);
    }

    // 11. Verify decryption
    if (memcmp(original_pixels, decrypted_pixels, pixel_data_len) == 0) {
        printf("Decryption verification: YES, original and decrypted images match.\n");
    }
    else {
        printf("Decryption verification: NO, original and decrypted images DO NOT match.\n");
    }

    // 12. Free all dynamically allocated memory
    stbi_image_free(original_pixels);
    free(encrypted_pixels);
    free(decrypted_pixels);

    return 0;
}