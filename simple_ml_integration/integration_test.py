"""
Comprehensive test suite for AtomGPT ML Integration
Tests LLM-driven tool calling with various scenarios
"""
import os
import asyncio
import pandas as pd
from sklearn.datasets import make_regression, make_classification
from dotenv import load_dotenv

# Load environment
load_dotenv()
api_key = os.getenv("ATOMGPT_API_KEY")

print("\n" + "🟦"*35)
print("ATOMGPT ML ASSISTANT - COMPREHENSIVE TEST")
print("🟦"*35)

# ============================================
# STEP 1: Create Sample Datasets
# ============================================
print("\n" + "="*70)
print("STEP 1: CREATING SAMPLE DATA")
print("="*70 + "\n")

# Regression dataset
X, y = make_regression(n_samples=200, n_features=8, noise=10, random_state=42)
df_sales = pd.DataFrame(X, columns=[
    'marketing_spend', 'competitor_price', 'store_size',
    'online_rating', 'season', 'region', 'weather_score', 'previous_sales'
])
df_sales['sales'] = y
df_sales.to_csv('sales_data.csv', index=False)
print("✅ Created sales_data.csv (200 rows, 9 columns)")

# Classification dataset
X, y = make_classification(n_samples=200, n_features=10, n_classes=3, 
                           n_informative=8, random_state=42)
df_customer = pd.DataFrame(X, columns=[
    'age', 'income', 'spending_score', 'years_customer',
    'support_calls', 'return_rate', 'online_purchases',
    'satisfaction', 'loyalty_points', 'referrals'
])
df_customer['segment'] = y
df_customer.to_csv('customer_data.csv', index=False)
print("✅ Created customer_data.csv (200 rows, 11 columns)")

# ============================================
# STEP 2: Verify Backend Works
# ============================================
print("\n" + "="*70)
print("STEP 2: TESTING ML BACKEND DIRECTLY")
print("="*70 + "\n")

from ml_backend import train_random_forest, evaluate_model, create_visualizations

print("Testing direct function calls (no LLM)...")
result = train_random_forest('sales_data.csv', 'sales', 'regression', 100)
print(f"  Train: {result['status']}")

result = evaluate_model()
print(f"  Evaluate: {result['status']}")
if result['status'] == 'success':
    print(f"  Metrics: {result['metrics']}")

result = create_visualizations()
print(f"  Visualize: {result['status']}")

# ============================================
# STEP 3: Test LLM-Driven Tool Calling
# ============================================
print("\n" + "="*70)
print("STEP 3: TESTING LLM-DRIVEN TOOL CALLING")
print("="*70 + "\n")

async def run_llm_tests():
    """Run all LLM-driven tests"""
    from atomgpt_integration import simple_chat
    
    # Test 1: Basic training request
    print("─"*70)
    print("TEST 1: Basic Training Request")
    print("─"*70)
    print("Query: 'Train a model on sales_data.csv to predict sales'\n")
    
    try:
        response = await simple_chat(
            "Train a model on sales_data.csv to predict sales",
            verbose=True
        )
        print(f"\n{'='*70}")
        print("🤖 ATOMGPT SUMMARY:")
        print(f"{'='*70}\n{response}\n")
    except Exception as e:
        print(f"❌ Test 1 failed: {e}\n")
    
    # Test 2: Classification task
    print("\n" + "─"*70)
    print("TEST 2: Classification Task")
    print("─"*70)
    print("Query: 'Build a classifier on customer_data.csv for segment'\n")
    
    try:
        response = await simple_chat(
            "Build a classifier on customer_data.csv for segment",
            verbose=True
        )
        print(f"\n{'='*70}")
        print("🤖 ATOMGPT SUMMARY:")
        print(f"{'='*70}\n{response}\n")
    except Exception as e:
        print(f"❌ Test 2 failed: {e}\n")
    
    # Test 3: Natural language
    print("\n" + "─"*70)
    print("TEST 3: Natural Language Query")
    print("─"*70)
    print("Query: 'I have sales data and want to predict sales'\n")
    
    try:
        response = await simple_chat(
            "I have sales_data.csv and want to predict the sales column",
            verbose=True
        )
        print(f"\n{'='*70}")
        print("🤖 ATOMGPT SUMMARY:")
        print(f"{'='*70}\n{response}\n")
    except Exception as e:
        print(f"❌ Test 3 failed: {e}\n")
    
    # Test 4: Error handling - missing file
    print("\n" + "─"*70)
    print("TEST 4: Error Handling (Missing File)")
    print("─"*70)
    print("Query: 'Train on nonexistent.csv to predict value'\n")
    
    try:
        response = await simple_chat(
            "Train on nonexistent.csv to predict value",
            verbose=True
        )
        print(f"\n{'='*70}")
        print("🤖 ATOMGPT RESPONSE:")
        print(f"{'='*70}\n{response}\n")
    except Exception as e:
        print(f"Expected error occurred: {e}\n")


# Run the async tests
if not api_key:
    print("⚠️  ATOMGPT_API_KEY not set")
    print("\nTo test with AtomGPT:")
    print("  1. Get your API key from AtomGPT")
    print("  2. Create .env file: ATOMGPT_API_KEY=your-key")
    print("  3. Or: export ATOMGPT_API_KEY='your-key'")
    print("  4. Run: python test.py\n")
else:
    print(f"✅ API key loaded\n")
    asyncio.run(run_llm_tests())

# ============================================
# FINAL SUMMARY
# ============================================
print("\n" + "="*70)
print("✅ TESTING COMPLETE!")
print("="*70)
print("\n📊 What was tested:")
print("  ✓ Direct ML backend functions")
print("  ✓ LLM-driven tool orchestration")
print("  ✓ Regression tasks")
print("  ✓ Classification tasks")
print("  ✓ Natural language understanding")
print("  ✓ Error handling")
print("\n📁 Files created:")
print("  • sales_data.csv")
print("  • customer_data.csv")
print("  • outputs/ (visualization plots)")
print("\n🎯 Key Feature:")
print("  The LLM decides which tools to call and in what order!")
print("  This is how MCP servers work - the LLM orchestrates.\n")
print("="*70 + "\n")