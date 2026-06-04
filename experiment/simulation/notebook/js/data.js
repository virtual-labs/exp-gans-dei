// DCGAN Training Experiment Data
// Digit-wise sequential training on MNIST digits 0-9
// Each digit trained separately for specified epochs

const EXPERIMENT_DATA = {
  "25": {
    epochs: 25,
    digitResults: {
      0: {
        trainingLog: `======================================================================
TRAINING DIGIT 0
======================================================================

Epoch [1/25] | G Loss: 1.550 | D Loss: 1.324 | Quality: ✅
Epoch [5/25] | G Loss: 1.335 | D Loss: 1.113 | Quality: ✅
Epoch [10/25] | G Loss: 0.951 | D Loss: 1.250 | Quality: ✅
Epoch [15/25] | G Loss: 1.005 | D Loss: 1.247 | Quality: ✅
Epoch [20/25] | G Loss: 0.927 | D Loss: 1.159 | Quality: ✅
Epoch [25/25] | G Loss: 0.938 | D Loss: 1.174 | Quality: ✅

✓ Digit 0 training complete!`,
        finalMetrics: { g_loss: 0.938, d_loss: 1.174 }
      },
      1: {
        trainingLog: `======================================================================
TRAINING DIGIT 1
======================================================================

Epoch [1/25] | G Loss: 1.925 | D Loss: 1.238 | Quality: ✅
Epoch [5/25] | G Loss: 1.406 | D Loss: 1.027 | Quality: ✅
Epoch [10/25] | G Loss: 0.952 | D Loss: 1.213 | Quality: ✅
Epoch [15/25] | G Loss: 0.882 | D Loss: 1.224 | Quality: ✅
Epoch [20/25] | G Loss: 0.915 | D Loss: 1.308 | Quality: ✅
Epoch [25/25] | G Loss: 0.921 | D Loss: 1.244 | Quality: ✅

✓ Digit 1 training complete!`,
        finalMetrics: { g_loss: 0.921, d_loss: 1.244 }
      },
      2: {
        trainingLog: `======================================================================
TRAINING DIGIT 2
======================================================================

Epoch [1/25] | G Loss: 1.469 | D Loss: 1.368 | Quality: ✅
Epoch [5/25] | G Loss: 1.310 | D Loss: 1.153 | Quality: ✅
Epoch [10/25] | G Loss: 0.926 | D Loss: 1.209 | Quality: ✅
Epoch [15/25] | G Loss: 0.948 | D Loss: 1.236 | Quality: ✅
Epoch [20/25] | G Loss: 0.955 | D Loss: 1.147 | Quality: ✅
Epoch [25/25] | G Loss: 0.974 | D Loss: 1.145 | Quality: ✅

✓ Digit 2 training complete!`,
        finalMetrics: { g_loss: 0.974, d_loss: 1.145 }
      },
      3: {
        trainingLog: `======================================================================
TRAINING DIGIT 3
======================================================================

Epoch [1/25] | G Loss: 1.654 | D Loss: 1.299 | Quality: ✅
Epoch [5/25] | G Loss: 1.294 | D Loss: 1.093 | Quality: ✅
Epoch [10/25] | G Loss: 0.895 | D Loss: 1.202 | Quality: ✅
Epoch [15/25] | G Loss: 0.935 | D Loss: 1.203 | Quality: ✅
Epoch [20/25] | G Loss: 0.958 | D Loss: 1.194 | Quality: ✅
Epoch [25/25] | G Loss: 0.957 | D Loss: 1.182 | Quality: ✅

✓ Digit 3 training complete!`,
        finalMetrics: { g_loss: 0.957, d_loss: 1.182 }
      },
      4: {
        trainingLog: `======================================================================
TRAINING DIGIT 4
======================================================================

Epoch [1/25] | G Loss: 1.672 | D Loss: 1.268 | Quality: ✅
Epoch [5/25] | G Loss: 1.332 | D Loss: 1.146 | Quality: ✅
Epoch [10/25] | G Loss: 0.909 | D Loss: 1.301 | Quality: ✅
Epoch [15/25] | G Loss: 0.963 | D Loss: 1.243 | Quality: ✅
Epoch [20/25] | G Loss: 0.934 | D Loss: 1.243 | Quality: ✅
Epoch [25/25] | G Loss: 0.956 | D Loss: 1.232 | Quality: ✅

✓ Digit 4 training complete!`,
        finalMetrics: { g_loss: 0.956, d_loss: 1.232 }
      },
      5: {
        trainingLog: `======================================================================
TRAINING DIGIT 5
======================================================================

Epoch [1/25] | G Loss: 1.572 | D Loss: 1.261 | Quality: ✅
Epoch [5/25] | G Loss: 1.270 | D Loss: 1.104 | Quality: ✅
Epoch [10/25] | G Loss: 0.992 | D Loss: 1.202 | Quality: ✅
Epoch [15/25] | G Loss: 1.005 | D Loss: 1.202 | Quality: ✅
Epoch [20/25] | G Loss: 0.996 | D Loss: 1.179 | Quality: ✅
Epoch [25/25] | G Loss: 1.011 | D Loss: 1.126 | Quality: ✅

✓ Digit 5 training complete!`,
        finalMetrics: { g_loss: 1.011, d_loss: 1.126 }
      },
      6: {
        trainingLog: `======================================================================
TRAINING DIGIT 6
======================================================================

Epoch [1/25] | G Loss: 1.611 | D Loss: 1.320 | Quality: ✅
Epoch [5/25] | G Loss: 1.286 | D Loss: 1.112 | Quality: ✅
Epoch [10/25] | G Loss: 0.946 | D Loss: 1.169 | Quality: ✅
Epoch [15/25] | G Loss: 0.950 | D Loss: 1.181 | Quality: ✅
Epoch [20/25] | G Loss: 0.977 | D Loss: 1.208 | Quality: ✅
Epoch [25/25] | G Loss: 0.971 | D Loss: 1.199 | Quality: ✅

✓ Digit 6 training complete!`,
        finalMetrics: { g_loss: 0.971, d_loss: 1.199 }
      },
      7: {
        trainingLog: `======================================================================
TRAINING DIGIT 7
======================================================================

Epoch [1/25] | G Loss: 1.965 | D Loss: 1.153 | Quality: ✅
Epoch [5/25] | G Loss: 1.259 | D Loss: 1.035 | Quality: ✅
Epoch [10/25] | G Loss: 1.018 | D Loss: 1.237 | Quality: ✅
Epoch [15/25] | G Loss: 0.928 | D Loss: 1.236 | Quality: ✅
Epoch [20/25] | G Loss: 0.912 | D Loss: 1.192 | Quality: ✅
Epoch [25/25] | G Loss: 0.951 | D Loss: 1.226 | Quality: ✅

✓ Digit 7 training complete!`,
        finalMetrics: { g_loss: 0.951, d_loss: 1.226 }
      },
      8: {
        trainingLog: `======================================================================
TRAINING DIGIT 8
======================================================================

Epoch [1/25] | G Loss: 1.677 | D Loss: 1.221 | Quality: ✅
Epoch [5/25] | G Loss: 1.428 | D Loss: 1.112 | Quality: ✅
Epoch [10/25] | G Loss: 0.889 | D Loss: 1.226 | Quality: ✅
Epoch [15/25] | G Loss: 0.921 | D Loss: 1.211 | Quality: ✅
Epoch [20/25] | G Loss: 0.961 | D Loss: 1.212 | Quality: ✅
Epoch [25/25] | G Loss: 0.968 | D Loss: 1.225 | Quality: ✅

✓ Digit 8 training complete!`,
        finalMetrics: { g_loss: 0.968, d_loss: 1.225 }
      },
      9: {
        trainingLog: `======================================================================
TRAINING DIGIT 9
======================================================================

Epoch [1/25] | G Loss: 1.598 | D Loss: 1.172 | Quality: ✅
Epoch [5/25] | G Loss: 1.406 | D Loss: 1.080 | Quality: ✅
Epoch [10/25] | G Loss: 0.972 | D Loss: 1.233 | Quality: ✅
Epoch [15/25] | G Loss: 0.943 | D Loss: 1.212 | Quality: ✅
Epoch [20/25] | G Loss: 0.942 | D Loss: 1.202 | Quality: ✅
Epoch [25/25] | G Loss: 0.977 | D Loss: 1.266 | Quality: ✅

✓ Digit 9 training complete!`,
        finalMetrics: { g_loss: 0.977, d_loss: 1.266 }
      }
    }
  },
  "30": {
    epochs: 30,
    digitResults: {
      0: {
        trainingLog: `======================================================================
TRAINING DIGIT 0
======================================================================

Epoch [1/30] | G Loss: 1.490 | D Loss: 1.431 | Quality: ✅
Epoch [5/30] | G Loss: 1.306 | D Loss: 1.083 | Quality: ✅
Epoch [10/30] | G Loss: 1.029 | D Loss: 1.236 | Quality: ✅
Epoch [15/30] | G Loss: 0.988 | D Loss: 1.175 | Quality: ✅
Epoch [20/30] | G Loss: 0.966 | D Loss: 1.124 | Quality: ✅
Epoch [25/30] | G Loss: 0.967 | D Loss: 1.232 | Quality: ✅
Epoch [30/30] | G Loss: 0.994 | D Loss: 1.258 | Quality: ✅

✓ Digit 0 training complete!`,
        finalMetrics: { g_loss: 0.994, d_loss: 1.258 }
      },
      1: {
        trainingLog: `======================================================================
TRAINING DIGIT 1
======================================================================

Epoch [1/30] | G Loss: 1.589 | D Loss: 1.312 | Quality: ✅
Epoch [5/30] | G Loss: 1.186 | D Loss: 1.177 | Quality: ✅
Epoch [10/30] | G Loss: 0.913 | D Loss: 1.309 | Quality: ✅
Epoch [15/30] | G Loss: 0.889 | D Loss: 1.264 | Quality: ✅
Epoch [20/30] | G Loss: 0.884 | D Loss: 1.263 | Quality: ✅
Epoch [25/30] | G Loss: 0.852 | D Loss: 1.293 | Quality: ✅
Epoch [30/30] | G Loss: 0.873 | D Loss: 1.349 | Quality: ✅

✓ Digit 1 training complete!`,
        finalMetrics: { g_loss: 0.873, d_loss: 1.349 }
      },
      2: {
        trainingLog: `======================================================================
TRAINING DIGIT 2
======================================================================

Epoch [1/30] | G Loss: 1.732 | D Loss: 1.266 | Quality: ✅
Epoch [5/30] | G Loss: 1.358 | D Loss: 1.056 | Quality: ✅
Epoch [10/30] | G Loss: 0.952 | D Loss: 1.242 | Quality: ✅
Epoch [15/30] | G Loss: 0.976 | D Loss: 1.174 | Quality: ✅
Epoch [20/30] | G Loss: 1.005 | D Loss: 1.219 | Quality: ✅
Epoch [25/30] | G Loss: 0.987 | D Loss: 1.144 | Quality: ✅
Epoch [30/30] | G Loss: 1.014 | D Loss: 1.163 | Quality: ✅

✓ Digit 2 training complete!`,
        finalMetrics: { g_loss: 1.014, d_loss: 1.163 }
      },
      3: {
        trainingLog: `======================================================================
TRAINING DIGIT 3
======================================================================

Epoch [1/30] | G Loss: 1.680 | D Loss: 1.379 | Quality: ✅
Epoch [5/30] | G Loss: 1.541 | D Loss: 1.188 | Quality: ✅
Epoch [10/30] | G Loss: 0.939 | D Loss: 1.304 | Quality: ✅
Epoch [15/30] | G Loss: 0.909 | D Loss: 1.238 | Quality: ✅
Epoch [20/30] | G Loss: 0.911 | D Loss: 1.198 | Quality: ✅
Epoch [25/30] | G Loss: 0.952 | D Loss: 1.194 | Quality: ✅
Epoch [30/30] | G Loss: 0.982 | D Loss: 1.215 | Quality: ✅

✓ Digit 3 training complete!`,
        finalMetrics: { g_loss: 0.982, d_loss: 1.215 }
      },
      4: {
        trainingLog: `======================================================================
TRAINING DIGIT 4
======================================================================

Epoch [1/30] | G Loss: 1.977 | D Loss: 1.106 | Quality: ✅
Epoch [5/30] | G Loss: 1.332 | D Loss: 1.087 | Quality: ✅
Epoch [10/30] | G Loss: 0.973 | D Loss: 1.263 | Quality: ✅
Epoch [15/30] | G Loss: 1.003 | D Loss: 1.217 | Quality: ✅
Epoch [20/30] | G Loss: 0.958 | D Loss: 1.174 | Quality: ✅
Epoch [25/30] | G Loss: 0.980 | D Loss: 1.160 | Quality: ✅
Epoch [30/30] | G Loss: 0.980 | D Loss: 1.115 | Quality: ✅

✓ Digit 4 training complete!`,
        finalMetrics: { g_loss: 0.980, d_loss: 1.115 }
      },
      5: {
        trainingLog: `======================================================================
TRAINING DIGIT 5
======================================================================

Epoch [1/30] | G Loss: 1.620 | D Loss: 1.415 | Quality: ✅
Epoch [5/30] | G Loss: 1.295 | D Loss: 1.096 | Quality: ✅
Epoch [10/30] | G Loss: 0.955 | D Loss: 1.194 | Quality: ✅
Epoch [15/30] | G Loss: 1.018 | D Loss: 1.133 | Quality: ✅
Epoch [20/30] | G Loss: 1.004 | D Loss: 1.131 | Quality: ✅
Epoch [25/30] | G Loss: 1.016 | D Loss: 1.142 | Quality: ✅
Epoch [30/30] | G Loss: 1.028 | D Loss: 1.176 | Quality: ✅

✓ Digit 5 training complete!`,
        finalMetrics: { g_loss: 1.028, d_loss: 1.176 }
      },
      6: {
        trainingLog: `======================================================================
TRAINING DIGIT 6
======================================================================

Epoch [1/30] | G Loss: 1.672 | D Loss: 1.259 | Quality: ✅
Epoch [5/30] | G Loss: 1.231 | D Loss: 1.105 | Quality: ✅
Epoch [10/30] | G Loss: 0.944 | D Loss: 1.163 | Quality: ✅
Epoch [15/30] | G Loss: 0.965 | D Loss: 1.188 | Quality: ✅
Epoch [20/30] | G Loss: 0.952 | D Loss: 1.165 | Quality: ✅
Epoch [25/30] | G Loss: 0.966 | D Loss: 1.150 | Quality: ✅
Epoch [30/30] | G Loss: 0.990 | D Loss: 1.139 | Quality: ✅

✓ Digit 6 training complete!`,
        finalMetrics: { g_loss: 0.990, d_loss: 1.139 }
      },
      7: {
        trainingLog: `======================================================================
TRAINING DIGIT 7
======================================================================

Epoch [1/30] | G Loss: 1.749 | D Loss: 1.265 | Quality: ✅
Epoch [5/30] | G Loss: 1.277 | D Loss: 1.114 | Quality: ✅
Epoch [10/30] | G Loss: 0.975 | D Loss: 1.251 | Quality: ✅
Epoch [15/30] | G Loss: 0.942 | D Loss: 1.267 | Quality: ✅
Epoch [20/30] | G Loss: 0.917 | D Loss: 1.183 | Quality: ✅
Epoch [25/30] | G Loss: 0.932 | D Loss: 1.226 | Quality: ✅
Epoch [30/30] | G Loss: 0.948 | D Loss: 1.209 | Quality: ✅

✓ Digit 7 training complete!`,
        finalMetrics: { g_loss: 0.948, d_loss: 1.209 }
      },
      8: {
        trainingLog: `======================================================================
TRAINING DIGIT 8
======================================================================

Epoch [1/30] | G Loss: 1.488 | D Loss: 1.285 | Quality: ✅
Epoch [5/30] | G Loss: 1.366 | D Loss: 1.152 | Quality: ✅
Epoch [10/30] | G Loss: 0.954 | D Loss: 1.263 | Quality: ✅
Epoch [15/30] | G Loss: 0.904 | D Loss: 1.193 | Quality: ✅
Epoch [20/30] | G Loss: 0.971 | D Loss: 1.211 | Quality: ✅
Epoch [25/30] | G Loss: 0.944 | D Loss: 1.167 | Quality: ✅
Epoch [30/30] | G Loss: 1.001 | D Loss: 1.213 | Quality: ✅

✓ Digit 8 training complete!`,
        finalMetrics: { g_loss: 1.001, d_loss: 1.213 }
      },
      9: {
        trainingLog: `======================================================================
TRAINING DIGIT 9
======================================================================

Epoch [1/30] | G Loss: 1.542 | D Loss: 1.234 | Quality: ✅
Epoch [5/30] | G Loss: 1.295 | D Loss: 1.156 | Quality: ✅
Epoch [10/30] | G Loss: 1.034 | D Loss: 1.262 | Quality: ✅
Epoch [15/30] | G Loss: 0.985 | D Loss: 1.269 | Quality: ✅
Epoch [20/30] | G Loss: 0.987 | D Loss: 1.223 | Quality: ✅
Epoch [25/30] | G Loss: 0.988 | D Loss: 1.171 | Quality: ✅
Epoch [30/30] | G Loss: 0.924 | D Loss: 1.164 | Quality: ✅

✓ Digit 9 training complete!`,
        finalMetrics: { g_loss: 0.924, d_loss: 1.164 }
      }
    }
  }
};
