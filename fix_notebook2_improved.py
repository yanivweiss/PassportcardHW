#!/usr/bin/env python
import json
import os
import traceback

def fix_notebook2():
    """Fix the model training issues with better diagnostics and output"""
    notebook_path = "notebooks/2_PassportCard_Model_Development.ipynb"
    backup_path = notebook_path + ".backup"
    
    print(f"🔍 Working directory: {os.getcwd()}")
    print(f"📝 Fixing notebook: {notebook_path}")
    
    if not os.path.exists(notebook_path):
        print(f"❌ ERROR: Notebook not found at {notebook_path}")
        return False
    
    print(f"📋 Creating backup at {backup_path}")
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook_content = f.read()
            
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(notebook_content)
            
        print(f"✅ Backup created successfully")
    except Exception as e:
        print(f"❌ ERROR creating backup: {str(e)}")
        traceback.print_exc()
        return False
            
    try:
        # Load the notebook as JSON
        with open(notebook_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
            
        print(f"✅ Loaded notebook JSON successfully")
        
        # Find and modify prepare_for_modeling function
        prepare_found = False
        evaluate_found = False
        
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                source = ''.join(cell['source'])
                
                # Fix the prepare_for_modeling function
                if 'def prepare_for_modeling' in source:
                    prepare_found = True
                    print(f"📌 Found prepare_for_modeling in cell {i}")
                    
                    # New function code with fixes
                    fixed_code = [
                        "def prepare_for_modeling(df, target_col='FutureClaimAmount', test_size=0.2, log_transform=True):\n",
                        "    \"\"\"Prepare data for modeling by splitting and transforming\"\"\"\n",
                        "    # Work with a copy\n",
                        "    data = df.copy()\n",
                        "    \n",
                        "    # Apply log transform to target if specified\n",
                        "    if log_transform:\n",
                        "        data['Log_' + target_col] = np.log1p(data[target_col])\n",
                        "        y_col = 'Log_' + target_col\n",
                        "    else:\n",
                        "        y_col = target_col\n",
                        "    \n",
                        "    # Select features and target\n",
                        "    feature_cols = [\n",
                        "        col for col in data.columns \n",
                        "        if col not in [target_col, 'Log_' + target_col, 'Member_ID', 'PolicyID']\n",
                        "    ]\n",
                        "    \n",
                        "    X = data[feature_cols]\n",
                        "    y = data[y_col]\n",
                        "    \n",
                        "    # Critical fix: Filter out non-numeric columns before model training\n",
                        "    print('Original X shape:', X.shape)\n",
                        "    date_cols = X.select_dtypes(include=['object', 'datetime64']).columns.tolist()\n",
                        "    print('Removing non-numeric columns:', date_cols)\n",
                        "    X = X.select_dtypes(include=['int64', 'float64'])\n",
                        "    print('New X shape after removing non-numeric columns:', X.shape)\n",
                        "    \n",
                        "    # Split data into training and testing sets\n",
                        "    X_train, X_test, y_train, y_test = train_test_split(\n",
                        "        X, y, test_size=test_size, random_state=42\n",
                        "    )\n",
                        "    \n",
                        "    # Scale numerical features\n",
                        "    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns\n",
                        "    scaler = RobustScaler()  # Robust to outliers\n",
                        "    \n",
                        "    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])\n",
                        "    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])\n",
                        "    \n",
                        "    return X_train, X_test, y_train, y_test, feature_cols, log_transform\n"
                    ]
                    
                    # Update the cell
                    cell['source'] = fixed_code
                    print(f"✅ Updated prepare_for_modeling function")
                
                # Fix the evaluate_model function
                if 'def evaluate_model(' in source:
                    evaluate_found = True
                    print(f"📌 Found evaluate_model in cell {i}")
                    
                    # New evaluation function with fixes
                    fixed_eval = [
                        "def evaluate_model(name, model, X_train, y_train, X_test, y_test, log_transform=True):\n",
                        "    \"\"\"Train and evaluate a model\"\"\"\n",
                        "    # Time training process\n",
                        "    start_time = time.time()\n",
                        "    \n",
                        "    # Ensure data types are numeric only (fail-safe check)\n",
                        "    print(f\"Checking {name} input data types...\")\n",
                        "    X_train = X_train.select_dtypes(include=['number'])\n",
                        "    X_test = X_test.select_dtypes(include=['number'])\n",
                        "    print(f\"X_train shape after type checking: {X_train.shape}\")\n",
                        "    \n",
                        "    # Train the model\n",
                        "    model.fit(X_train, y_train)\n",
                        "    \n",
                        "    # Calculate training time\n",
                        "    train_time = time.time() - start_time\n",
                        "    \n",
                        "    # Make predictions\n",
                        "    y_pred_train = model.predict(X_train)\n",
                        "    y_pred_test = model.predict(X_test)\n",
                        "    \n",
                        "    # Transform predictions back to original scale if log-transformed\n",
                        "    if log_transform:\n",
                        "        y_pred_train_orig = np.expm1(y_pred_train)\n",
                        "        y_pred_test_orig = np.expm1(y_pred_test)\n",
                        "        y_train_orig = np.expm1(y_train)\n",
                        "        y_test_orig = np.expm1(y_test)\n",
                        "    else:\n",
                        "        y_pred_train_orig = y_pred_train\n",
                        "        y_pred_test_orig = y_pred_test\n",
                        "        y_train_orig = y_train\n",
                        "        y_test_orig = y_test\n",
                        "    \n",
                        "    # Calculate metrics\n",
                        "    rmse_train = np.sqrt(mean_squared_error(y_train_orig, y_pred_train_orig))\n",
                        "    rmse_test = np.sqrt(mean_squared_error(y_test_orig, y_pred_test_orig))\n",
                        "    \n",
                        "    mae_train = mean_absolute_error(y_train_orig, y_pred_train_orig)\n",
                        "    mae_test = mean_absolute_error(y_test_orig, y_pred_test_orig)\n",
                        "    \n",
                        "    r2_train = r2_score(y_train_orig, y_pred_train_orig)\n",
                        "    r2_test = r2_score(y_test_orig, y_pred_test_orig)\n",
                        "    \n",
                        "    # Calculate MAPE (Mean Absolute Percentage Error) for values > 10\n",
                        "    # to avoid division by very small values\n",
                        "    train_idx = y_train_orig > 10\n",
                        "    test_idx = y_test_orig > 10\n",
                        "    \n",
                        "    if any(train_idx):\n",
                        "        mape_train = np.mean(np.abs((y_train_orig[train_idx] - y_pred_train_orig[train_idx]) / y_train_orig[train_idx])) * 100\n",
                        "    else:\n",
                        "        mape_train = np.nan\n",
                        "        \n",
                        "    if any(test_idx):\n",
                        "        mape_test = np.mean(np.abs((y_test_orig[test_idx] - y_pred_test_orig[test_idx]) / y_test_orig[test_idx])) * 100\n",
                        "    else:\n",
                        "        mape_test = np.nan\n",
                        "    \n",
                        "    # Organize results\n",
                        "    results = {\n",
                        "        'Model': name,\n",
                        "        'RMSE_Train': rmse_train,\n",
                        "        'RMSE_Test': rmse_test,\n",
                        "        'MAE_Train': mae_train,\n",
                        "        'MAE_Test': mae_test,\n",
                        "        'R2_Train': r2_train,\n",
                        "        'R2_Test': r2_test,\n",
                        "        'MAPE_Train': mape_train,\n",
                        "        'MAPE_Test': mape_test,\n",
                        "        'Training_Time': train_time\n",
                        "    }\n",
                        "    \n",
                        "    return results, model, y_pred_test_orig\n"
                    ]
                    
                    # Update the cell
                    cell['source'] = fixed_eval
                    print(f"✅ Updated evaluate_model function")
        
        if not prepare_found:
            print(f"❌ WARNING: Could not find prepare_for_modeling function in the notebook!")
        
        if not evaluate_found:
            print(f"❌ WARNING: Could not find evaluate_model function in the notebook!")
        
        # Save the modified notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1)
            
        print(f"✅ Successfully saved fixed notebook to {notebook_path}")
        print("\n📋 NEXT STEPS:")
        print("1. In Jupyter, restart the kernel for notebook 2")
        print("2. Run all cells from the beginning")
        return True
        
    except Exception as e:
        print(f"❌ ERROR during notebook modification: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 IMPROVED NOTEBOOK 2 FIXER 🔧")
    print("===============================")
    fix_notebook2() 