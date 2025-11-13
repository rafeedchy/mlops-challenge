#!/usr/bin/env bash
# ==========================================================
# Script: test_predictions.sh
# Purpose: Send 15 test prediction requests to the Iris API
# ==========================================================

URL="http://127.0.0.1:8000/predict"
HEADER="Content-Type: application/json"

echo "🌸 Starting test predictions against $URL ..."
echo "------------------------------------------------------------"

# 1
echo "🔹 [1] Classic Setosa"
curl -s -X POST $URL -H "$HEADER" -d '{"instances": [[5.1,3.5,1.4,0.2]]}'
echo -e "\n------------------------------------------------------------"; sleep 1

# 2
echo "🔹 [2] Slightly wider sepal (Setosa)"
curl -s -X POST $URL -H "$HEADER" -d '{"instances": [[4.9,3.6,1.4,0.1]]}'
echo -e "\n------------------------------------------------------------"; sleep 1

# 3
echo "🔹 [3] Compact Setosa"
curl -s -X POST $URL -H "$HEADER" -d '{"instances": [[4.7,3.2,1.3,0.2]]}'
echo -e "\n------------------------------------------------------------"; sleep 1

# 4
echo "🔹 [4] Versicolor region"
curl -s -X POST $URL -H "$HEADER" -d '{"instances": [[6.0,2.2,4.0,1.0]]}'
echo -e "\n------------------------------------------------------------"; sleep 1

# 5
echo "🔹 [5] Classic Versicolor"
curl -s -X POST $URL -H "$HEADER" -d '{"instances": [[6.4,3.2,4.5,1.5]]}'
echo -e "\n------------------------------------------------------------"; sleep 1

# 6
echo "🔹 [6] Near border Versicolor/Virginica"
curl -s -X POST $URL -H "$HEADER" -d '{"instances": [[6.2,2.8,4.8,1.8]]}'
echo -e "\n------------------------------------------------------------"; sleep 1

# 7
echo "🔹 [7] Virginica (long petals)"
curl -s -X POST $URL -H "$HEADER" -d '{"instances": [[7.1,3.0,5.9,2.1]]}'
echo -e "\n------------------------------------------------------------"; sleep 1

# 8
echo "🔹 [8] Very large Virginica"
curl -s -X POST $URL -H "$HEADER" -d '{"instances": [[7.9,3.8,6.4,2.0]]}'
echo -e "\n------------------------------------------------------------"; sleep 1

# 9
echo "🔹 [9] Narrow sepal Virginica"
curl -s -X POST $URL -H "$HEADER" -d '{"instances": [[6.5,3.0,5.5,1.8]]}'
echo -e "\n------------------------------------------------------------"; sleep 1

# 10
echo "🔹 [10] Batch of 3 samples"
curl -s -X POST $URL -H "$HEADER" -d '{"instances": [[5.1,3.5,1.4,0.2],[6.7,3.1,4.7,1.5],[7.6,3.0,6.6,2.1]]}'
echo -e "\n------------------------------------------------------------"; sleep 1

# 11
echo "🔹 [11] Perturbed Setosa (drift test)"
curl -s -X POST $URL -H "$HEADER" -d '{"instances": [[5.0,3.0,1.6,0.3]]}'
echo -e "\n------------------------------------------------------------"; sleep 1

# 12
echo "🔹 [12] Smaller Versicolor"
curl -s -X POST $URL -H "$HEADER" -d '{"instances": [[5.9,3.0,4.2,1.5]]}'
echo -e "\n------------------------------------------------------------"; sleep 1

# 13
echo "🔹 [13] Smaller sepal Virginica"
curl -s -X POST $URL -H "$HEADER" -d '{"instances": [[6.3,2.9,5.6,1.8]]}'
echo -e "\n------------------------------------------------------------"; sleep 1

# 14
echo "🔹 [14] Edge case - longer petals"
curl -s -X POST $URL -H "$HEADER" -d '{"instances": [[7.2,3.2,6.0,2.2]]}'
echo -e "\n------------------------------------------------------------"; sleep 1

# 15
echo "🔹 [15] Mixed drift scenario"
curl -s -X POST $URL -H "$HEADER" -d '{"instances": [[5.4,3.9,1.7,0.4],[6.9,3.1,4.9,1.5],[7.7,2.6,6.9,2.3]]}'
echo -e "\n------------------------------------------------------------"

echo "✅ All 15 test predictions completed!"
