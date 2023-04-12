#define FLOATTYPE float

// below used for reference impl
// #define FH 21
// #define FW 21

#define TW 32
#define TH 32

#define HS 1
#define WS 1

// Max 1024 Threads per Block
#define BH 32
#define BW 32

#define DIV_RUP(x, y) ((x + y - 1) / y)

#define indexToOffset(x, y, channel, H, W, heightOffset, widthOffset) ((channel * H * W) + (heightOffset + y) * W + widthOffset + x)

#define pixel_x(blockWidth, blockWidthOffset, x) ((blockWidth * blockWidthOffset) + x)
#define pixel_y(blockHeight, blockHeightOffset, y) ((blockHeight * blockHeightOffset) + y)

#define shmem_offset(x_offset, y_offset, x, y, pTW, pw, ph) (((y_offset + y + ph) * pTW + (x_offset + x + pw)))

#define CUDA_CALL(x)                   \
    do                                 \
    {                                  \
        cudaError_t ____rc = (x);      \
        assert(____rc == cudaSuccess); \
    } while (0)
