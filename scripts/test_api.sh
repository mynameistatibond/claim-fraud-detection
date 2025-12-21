#!/bin/bash

# Fraud Detection API - Example curl Commands

BASE_URL="http://localhost:8000"

echo "========================================="
echo "Fraud Detection API - Example Requests"
echo "========================================="
echo ""

# Test 1: Dashboard Scenario (Calibrated) with XGBoost
echo "1. Dashboard Scenario (Calibrated XGBoost):"
echo "-----------------------------------------"
curl -X POST "${BASE_URL}/predict?model=xgb&scenario=dashboard" \
  -H "Content-Type: application/json" \
  -d @../example_claim.json
echo -e "\n\n"

# Test 2: Auto-Flagger Scenario (Uncalibrated) with RandomForest
echo "2. Auto-Flagger Scenario (Uncalibrated RandomForest):"
echo "-----------------------------------------------------"
curl -X POST "${BASE_URL}/predict?model=rf&scenario=auto_flagger" \
  -H "Content-Type: application/json" \
  -d @../example_claim.json
echo -e "\n\n"

# Test 3: High-Risk Claim Example
echo "3. High-Risk Claim (Auto-Flagger with ExtraTrees):"
echo "---------------------------------------------------"
curl -X POST "${BASE_URL}/predict?model=et&scenario=auto_flagger" \
  -H "Content-Type: application/json" \
  -d '{
    "policy_annual_premium": 500,
    "total_claim_amount": 50000,
    "vehicle_age": 1,
    "days_since_bind": 10,
    "months_as_customer": 2,
    "capital-gains": 10000,
    "capital-loss": 0,
    "injury_share": 0.8,
    "property_share": 0.2,
    "umbrella_limit": 0,
    "incident_hour_of_the_day": 3
  }'
echo -e "\n\n"

# Test 4: Health Check
echo "4. Health Check:"
echo "----------------"
curl -X GET "${BASE_URL}/health"
echo -e "\n\n"

echo "========================================="
echo "All tests completed!"
echo "========================================="
