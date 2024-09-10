#define verbosePrintf(fmt,...) \
    do { if (VERBOSE) fprintf(stderr, "[VERBOSE] %s:%d:%s(): " fmt, __FILE__, \
                              __LINE__, __func__, ##__VA_ARGS__); } while (0)
#define debugPrintf(fmt,...) \
    do { if (DEBUG) fprintf(stderr, "[DEBUG] %s:%d:%s(): " fmt, __FILE__, \
                            __LINE__, __func__, ##__VA_ARGS__); } while (0)
#define infoPrintf(fmt,...) \
    do { if (INFO) fprintf(stderr, "[INFO] %s:%d:%s(): " fmt, __FILE__, \
                            __LINE__, __func__, ##__VA_ARGS__); } while (0)

#define CANTIDAD_ITERACIONES 100
#define MEASURES 10
#define OFFSET -0.4999f

#define NCORTE 1024
#define ALPHA 0.75
