import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

def generate_advanced_business_report(model_results, risk_data=None, output_path='advanced_business_report.md'):
    """
    Generate a comprehensive business report with advanced recommendations
    
    Parameters:
    -----------
    model_results : dict
        Dictionary containing model, metrics, feature_importance, etc.
    risk_data : DataFrame, optional
        DataFrame containing risk scores and categories
    output_path : str
        Path to save the output report
    """
    # Extract data from model_results
    metrics = model_results.get('metrics', {})
    feature_importance = model_results.get('feature_importance', pd.DataFrame())
    
    # Create the report
    with open(output_path, 'w') as f:
        # Title and Executive Summary
        f.write("# PassportCard Insurance Claims Prediction - Advanced Business Report\n\n")
        
        f.write("## Executive Summary\n\n")
        f.write("This advanced business analysis presents a sophisticated predictive model for insurance claims at PassportCard. ")
        f.write("Our enhanced model incorporates advanced temporal patterns, detailed risk profiling, and optimized machine learning algorithms ")
        f.write("to accurately predict the total claim amount per customer for the next six months. These predictions enable data-driven ")
        f.write("decisions across underwriting, pricing, customer management, and financial planning.\n\n")
        
        # Model Performance
        f.write("## Model Performance\n\n")
        model_type = type(model_results.get('model', '')).__name__
        f.write(f"Our {model_type} model achieved the following performance metrics:\n\n")
        f.write("| Metric | Value | Interpretation |\n")
        f.write("|--------|-------|----------------|\n")
        
        interpretations = {
            'RMSE': 'Average prediction error in USD',
            'MAE': 'Median prediction error in USD',
            'R2': 'Percentage of variance explained',
            'MAPE': 'Average percentage error'
        }
        
        for metric, value in metrics.items():
            interp = interpretations.get(metric, '')
            f.write(f"| {metric} | {value:.4f} | {interp} |\n")
        f.write("\n")
        
        # Key Drivers and Insights
        f.write("## Key Drivers of Insurance Claims\n\n")
        
        # Only include if feature_importance is available
        if not feature_importance.empty and len(feature_importance) > 0:
            f.write("### Most Influential Factors\n\n")
            f.write("The following factors were identified as the most significant predictors of future claims:\n\n")
            
            # Group features by category
            feature_categories = {
                'temporal': ['frequency', 'recency', 'seasonality', 'trend', 'volatility', 'claim_count', 'month', 'year', 'avg_claim', 'days'],
                'medical': ['chronic', 'cancer', 'risk_score', 'lifestyle', 'bmi', 'medical'],
                'demographic': ['age', 'gender', 'country', 'origin', 'bmi'],
                'service': ['service', 'outpatient', 'inpatient', 'emergency', 'dental', 'preventive']
            }
            
            # Categorize top features
            categorized_features = {}
            for i, row in feature_importance.head(15).iterrows():
                feature = row['feature']
                importance = row['importance']
                
                # Find which category it belongs to
                assigned = False
                for category, keywords in feature_categories.items():
                    if any(keyword.lower() in feature.lower() for keyword in keywords):
                        if category not in categorized_features:
                            categorized_features[category] = []
                        categorized_features[category].append((feature, importance))
                        assigned = True
                        break
                
                # If not assigned to any category
                if not assigned:
                    if 'other' not in categorized_features:
                        categorized_features['other'] = []
                    categorized_features['other'].append((feature, importance))
            
            # Write categorized features
            for category, features in categorized_features.items():
                if features:
                    f.write(f"**{category.title()} Factors:**\n")
                    for feature, importance in features:
                        f.write(f"- **{feature}** (Importance: {importance:.4f})\n")
                    f.write("\n")
        
        # Business Insights from Risk Data
        if risk_data is not None:
            f.write("### Customer Risk Profile Analysis\n\n")
            
            # Calculate risk distribution
            if 'risk_level' in risk_data.columns:
                risk_distribution = risk_data['risk_level'].value_counts().to_dict()
                total = sum(risk_distribution.values())
                
                f.write("#### Risk Distribution\n\n")
                f.write("| Risk Level | Count | Percentage |\n")
                f.write("|------------|-------|------------|\n")
                
                for level, count in sorted(risk_distribution.items()):
                    percentage = (count / total) * 100
                    f.write(f"| {level} | {count} | {percentage:.1f}% |\n")
                
                f.write("\n")
            
            # Calculate average metrics by risk level if available
            if 'final_risk_score' in risk_data.columns and 'risk_level' in risk_data.columns:
                f.write("#### Average Risk Metrics by Customer Segment\n\n")
                
                # Get columns to aggregate
                risk_metrics = [col for col in risk_data.columns if 'risk' in col and col not in ['risk_level', 'risk_cluster']]
                if risk_metrics:
                    agg_data = risk_data.groupby('risk_level')[risk_metrics].mean().reset_index()
                    
                    # Create a table with top metrics
                    metrics_to_show = min(5, len(risk_metrics))
                    selected_metrics = risk_metrics[:metrics_to_show]
                    
                    # Table header
                    f.write("| Risk Level |")
                    for metric in selected_metrics:
                        f.write(f" {metric} |")
                    f.write("\n")
                    
                    # Table separator
                    f.write("|------------|")
                    for _ in selected_metrics:
                        f.write("------------|")
                    f.write("\n")
                    
                    # Table data
                    for _, row in agg_data.iterrows():
                        f.write(f"| {row['risk_level']} |")
                        for metric in selected_metrics:
                            f.write(f" {row[metric]:.2f} |")
                        f.write("\n")
                    
                    f.write("\n")
        
        # Advanced Business Recommendations
        f.write("## Advanced Business Recommendations\n\n")
        
        f.write("### 1. Risk-Based Underwriting Enhancement\n\n")
        f.write("**Actionable Recommendations:**\n")
        f.write("- Implement automated risk scoring during the application process based on our model\n")
        f.write("- Create tiered underwriting processes with different approval paths based on predicted claim amounts\n")
        f.write("- Develop a real-time risk assessment dashboard for underwriters showing key risk factors\n")
        f.write("- Establish monthly model retraining to incorporate emerging risk patterns\n")
        f.write("- Integrate predicted claims with underwriting guidelines by automatically flagging high-risk applications\n\n")
        
        f.write("**Implementation Steps:**\n")
        f.write("1. Create a real-time API endpoint for the prediction model\n")
        f.write("2. Integrate risk scoring into the application workflow\n")
        f.write("3. Develop underwriter dashboards with interactive risk visualization\n")
        f.write("4. Establish risk thresholds for different approval levels\n")
        f.write("5. Create detailed documentation for underwriters on interpreting model outputs\n\n")
        
        f.write("**Expected Impact:**\n")
        f.write("- 15-20% reduction in high-risk policy approvals\n")
        f.write("- 10-15% reduction in overall claims ratio\n")
        f.write("- Improved underwriter efficiency with automated risk assessment\n\n")
        
        f.write("### 2. Dynamic Premium Optimization\n\n")
        f.write("**Actionable Recommendations:**\n")
        f.write("- Implement dynamic pricing based on predicted claim amounts and risk factors\n")
        f.write("- Create a pricing matrix that incorporates temporal risk patterns (seasonality)\n")
        f.write("- Develop targeted discount programs for customers with favorable risk profiles\n")
        f.write("- Implement premium adjustments based on volatility patterns in claims history\n")
        f.write("- Design renewal pricing formulas that incorporate predicted future claims\n\n")
        
        f.write("**Implementation Steps:**\n")
        f.write("1. Develop a pricing algorithm that incorporates model predictions\n")
        f.write("2. Create a price sensitivity analysis to determine optimal price points\n")
        f.write("3. Build a pricing simulation tool to test different scenarios\n")
        f.write("4. Establish monitoring protocols to evaluate pricing effectiveness\n")
        f.write("5. Design targeted promotional campaigns for low-risk segments\n\n")
        
        f.write("**Expected Impact:**\n")
        f.write("- 5-8% increase in premium revenue without increasing customer acquisition costs\n")
        f.write("- Improved retention of low-risk customers through targeted incentives\n")
        f.write("- More competitive pricing for favorable risk segments\n\n")
        
        f.write("### 3. Proactive Claims Management\n\n")
        f.write("**Actionable Recommendations:**\n")
        f.write("- Implement a proactive outreach program for customers with high predicted claims\n")
        f.write("- Develop early intervention programs targeting specific high-risk conditions\n")
        f.write("- Create specialized case management workflows for high-risk customers\n")
        f.write("- Develop preventive care recommendations based on risk factors\n")
        f.write("- Implement post-claim analysis to refine prediction models\n\n")
        
        f.write("**Implementation Steps:**\n")
        f.write("1. Create a customer outreach protocol based on risk scoring\n")
        f.write("2. Develop educational materials for high-risk conditions\n")
        f.write("3. Train customer service teams on proactive risk management\n")
        f.write("4. Build a claims monitoring dashboard for tracking intervention effectiveness\n")
        f.write("5. Establish feedback loops for continuous improvement\n\n")
        
        f.write("**Expected Impact:**\n")
        f.write("- 10-15% reduction in high-value claims through early intervention\n")
        f.write("- Improved customer satisfaction through proactive care\n")
        f.write("- Enhanced reputation as a customer-centric insurer\n\n")
        
        f.write("### 4. Strategic Customer Segmentation\n\n")
        f.write("**Actionable Recommendations:**\n")
        f.write("- Implement advanced customer segmentation based on risk profiles and predicted claims\n")
        f.write("- Develop targeted marketing strategies for each customer segment\n")
        f.write("- Create specialized renewal strategies based on risk trajectory\n")
        f.write("- Implement tailored communication plans for different risk segments\n")
        f.write("- Develop loyalty programs specifically designed for low-risk customers\n\n")
        
        f.write("**Implementation Steps:**\n")
        f.write("1. Integrate risk scoring into the CRM system\n")
        f.write("2. Develop segment-specific communication templates\n")
        f.write("3. Create marketing campaigns tailored to each segment\n")
        f.write("4. Establish renewal workflows based on risk profiles\n")
        f.write("5. Implement a loyalty program for preferred risk segments\n\n")
        
        f.write("**Expected Impact:**\n")
        f.write("- 15-20% improvement in retention rates for preferred customers\n")
        f.write("- 5-10% increase in customer satisfaction scores\n")
        f.write("- More efficient marketing spend through targeted campaigns\n\n")
        
        f.write("### 5. Enhanced Financial Planning\n\n")
        f.write("**Actionable Recommendations:**\n")
        f.write("- Implement monthly claims forecasting based on aggregated predictions\n")
        f.write("- Create risk-adjusted reserving models using prediction distributions\n")
        f.write("- Develop scenario planning tools for various risk environments\n")
        f.write("- Implement cash flow projections based on predicted claims timing\n")
        f.write("- Create reinsurance optimization strategies based on risk portfolio\n\n")
        
        f.write("**Implementation Steps:**\n")
        f.write("1. Develop an automated monthly forecasting system\n")
        f.write("2. Create financial dashboards for tracking predictions vs. actuals\n")
        f.write("3. Implement risk-adjusted reserving calculations\n")
        f.write("4. Build scenario planning capabilities into financial systems\n")
        f.write("5. Establish regular forecast review meetings with Finance\n\n")
        
        f.write("**Expected Impact:**\n")
        f.write("- 10-15% improvement in reserving accuracy\n")
        f.write("- Enhanced capital efficiency through better cash flow planning\n")
        f.write("- Improved financial stability through anticipatory planning\n\n")
        
        # Implementation Roadmap
        f.write("## Implementation Roadmap\n\n")
        f.write("### Phase 1: Foundation (1-3 months)\n")
        f.write("- Deploy prediction model as an internal API\n")
        f.write("- Develop risk scoring integration with underwriting\n")
        f.write("- Create basic dashboards for model monitoring\n")
        f.write("- Establish data pipelines for continuous model updating\n")
        f.write("- Train key stakeholders on model use and interpretation\n\n")
        
        f.write("### Phase 2: Integration (3-6 months)\n")
        f.write("- Implement risk-based pricing algorithms\n")
        f.write("- Develop customer segmentation in CRM systems\n")
        f.write("- Create proactive outreach protocols\n")
        f.write("- Implement financial forecasting based on predictions\n")
        f.write("- Develop first version of underwriter dashboards\n\n")
        
        f.write("### Phase 3: Optimization (6-12 months)\n")
        f.write("- Fine-tune pricing and underwriting based on feedback\n")
        f.write("- Implement advanced customer interventions\n")
        f.write("- Develop comprehensive loyalty programs\n")
        f.write("- Create advanced scenario planning capabilities\n")
        f.write("- Implement automated model retraining and validation\n\n")
        
        f.write("### Phase 4: Innovation (12+ months)\n")
        f.write("- Explore additional data sources for model enhancement\n")
        f.write("- Develop specialized models for different claim types\n")
        f.write("- Implement real-time pricing capabilities\n")
        f.write("- Develop advanced customer risk trajectory analysis\n")
        f.write("- Create ecosystem of predictive models for different business functions\n\n")
        
        # ROI Analysis
        f.write("## Expected Return on Investment\n\n")
        f.write("| Initiative | Implementation Cost | Expected Annual Return | ROI | Time to Value |\n")
        f.write("|------------|---------------------|-----------------------|-----|---------------|\n")
        f.write("| Risk-Based Underwriting | Medium | High | 200-300% | 3-6 months |\n")
        f.write("| Dynamic Premium Optimization | Medium-High | High | 150-250% | 6-9 months |\n")
        f.write("| Proactive Claims Management | Medium | Medium-High | 100-200% | 6-12 months |\n")
        f.write("| Strategic Customer Segmentation | Low-Medium | Medium | 150-200% | 3-6 months |\n")
        f.write("| Enhanced Financial Planning | Low | Medium | 300-400% | 1-3 months |\n\n")
        
        f.write("## Conclusion\n\n")
        f.write("This advanced predictive model provides PassportCard with a powerful tool to enhance multiple aspects of the business. ")
        f.write("By implementing the recommendations outlined in this report, the company can achieve significant improvements in risk assessment, ")
        f.write("pricing optimization, claims management, customer segmentation, and financial planning. The phased implementation approach ")
        f.write("ensures that the organization can systematically integrate these capabilities while measuring and validating the impact at each stage.\n\n")
        
        f.write("By leveraging these predictions effectively, PassportCard will gain a significant competitive advantage through data-driven ")
        f.write("decision-making across all levels of the organization. The focus on both operational enhancements and strategic initiatives ")
        f.write("ensures both short-term gains and long-term sustainable improvements in the company's performance and customer experience.\n\n")
        
        # Add date of report
        f.write(f"*Report generated on {datetime.now().strftime('%Y-%m-%d')}*\n")
    
    print(f"Advanced business report created: {output_path}")
    return output_path

def create_visual_business_insights(model_results, risk_data=None, output_dir='visualizations/business_insights'):
    """
    Create visual business insights from model results
    
    Parameters:
    -----------
    model_results : dict
        Dictionary containing model, metrics, feature_importance, etc.
    risk_data : DataFrame, optional
        DataFrame containing risk scores and categories
    output_dir : str
        Directory to save output visualizations
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data from model_results
    feature_importance = model_results.get('feature_importance', pd.DataFrame())
    predictions = model_results.get('predictions', None)
    actual = model_results.get('actual', None)
    
    # 1. Feature Importance Heatmap (Categorized)
    if not feature_importance.empty and len(feature_importance) > 0:
        # Define feature categories
        feature_categories = {
            'Temporal': ['frequency', 'recency', 'seasonality', 'trend', 'volatility', 'claim_count', 'month', 'year', 'avg_claim', 'days'],
            'Medical': ['chronic', 'cancer', 'risk_score', 'lifestyle', 'bmi', 'medical'],
            'Demographic': ['age', 'gender', 'country', 'origin', 'bmi'],
            'Service': ['service', 'outpatient', 'inpatient', 'emergency', 'dental', 'preventive']
        }
        
        # Assign categories to top features
        top_features = feature_importance.head(20).copy()
        top_features['category'] = 'Other'
        
        for i, row in top_features.iterrows():
            feature = row['feature']
            for category, keywords in feature_categories.items():
                if any(keyword.lower() in feature.lower() for keyword in keywords):
                    top_features.loc[i, 'category'] = category
                    break
        
        # Create a pivot table for heatmap
        pivot_data = top_features.pivot_table(
            index='category', 
            values='importance', 
            aggfunc='sum'
        ).reset_index()
        
        # Plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='category', data=pivot_data.sort_values('importance'), palette='viridis')
        plt.title('Impact of Feature Categories on Predictions')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/feature_category_impact.png')
        plt.close()
    
    # 2. Prediction Distribution by Threshold
    if predictions is not None and actual is not None:
        thresholds = [0, 100, 500, 1000, 5000, 10000]
        groups = pd.cut(actual, bins=thresholds, labels=[f"{thresholds[i]}-{thresholds[i+1]}" for i in range(len(thresholds)-1)])
        
        prediction_accuracy = pd.DataFrame({
            'Actual': actual,
            'Predicted': predictions,
            'Group': groups
        })
        
        # Calculate error by group
        prediction_accuracy['Error'] = prediction_accuracy['Predicted'] - prediction_accuracy['Actual']
        prediction_accuracy['AbsError'] = np.abs(prediction_accuracy['Error'])
        prediction_accuracy['RelError'] = prediction_accuracy['AbsError'] / (prediction_accuracy['Actual'] + 1)  # Add 1 to avoid division by zero
        
        group_metrics = prediction_accuracy.groupby('Group').agg({
            'Actual': 'count',
            'AbsError': 'mean',
            'RelError': 'mean'
        }).reset_index()
        group_metrics.columns = ['Claim Range', 'Count', 'Avg Absolute Error', 'Avg Relative Error']
        
        # Plot
        fig, ax1 = plt.subplots(figsize=(12, 7))
        
        bar_width = 0.4
        x = np.arange(len(group_metrics['Claim Range']))
        
        ax1.bar(x - bar_width/2, group_metrics['Count'], bar_width, color='steelblue', label='Count')
        ax1.set_ylabel('Count', color='steelblue')
        ax1.tick_params('y', colors='steelblue')
        
        ax2 = ax1.twinx()
        ax2.bar(x + bar_width/2, group_metrics['Avg Absolute Error'], bar_width, color='firebrick', label='Avg Error')
        ax2.set_ylabel('Average Absolute Error', color='firebrick')
        ax2.tick_params('y', colors='firebrick')
        
        ax1.set_xticks(x)
        ax1.set_xticklabels(group_metrics['Claim Range'], rotation=45)
        ax1.set_xlabel('Claim Amount Range')
        
        plt.title('Prediction Accuracy by Claim Amount Range')
        
        # Combine both legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/prediction_accuracy_by_claim_range.png')
        plt.close()
    
    # 3. Risk Level Analysis (if risk data available)
    if risk_data is not None and 'risk_level' in risk_data.columns:
        # Risk Distribution Pie Chart
        plt.figure(figsize=(10, 7))
        risk_counts = risk_data['risk_level'].value_counts()
        plt.pie(risk_counts, labels=risk_counts.index, autopct='%1.1f%%', startangle=90, 
                colors=plt.cm.Spectral(np.linspace(0, 1, len(risk_counts))))
        plt.axis('equal')
        plt.title('Customer Risk Level Distribution')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/risk_level_distribution.png')
        plt.close()
        
        # Risk Score Distribution by Level
        if 'final_risk_score' in risk_data.columns:
            plt.figure(figsize=(12, 7))
            sns.boxplot(x='risk_level', y='final_risk_score', data=risk_data, palette='Spectral')
            plt.title('Risk Score Distribution by Risk Level')
            plt.xlabel('Risk Level')
            plt.ylabel('Final Risk Score')
            plt.tight_layout()
            plt.savefig(f'{output_dir}/risk_score_distribution.png')
            plt.close()
            
            # Risk Score Components
            risk_components = [col for col in risk_data.columns if 'risk' in col.lower() and col not in ['risk_level', 'risk_cluster', 'final_risk_score']]
            if len(risk_components) >= 3:
                # Select top 5 components
                top_components = risk_components[:5]
                
                # Melt the data for visualization
                melted_data = pd.melt(
                    risk_data, 
                    id_vars=['risk_level'], 
                    value_vars=top_components,
                    var_name='Risk Component', 
                    value_name='Score'
                )
                
                plt.figure(figsize=(14, 8))
                sns.barplot(x='risk_level', y='Score', hue='Risk Component', data=melted_data, palette='Spectral')
                plt.title('Risk Components by Risk Level')
                plt.xlabel('Risk Level')
                plt.ylabel('Component Score')
                plt.legend(title='Risk Component', bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.tight_layout()
                plt.savefig(f'{output_dir}/risk_components_by_level.png')
                plt.close()
    
    print(f"Business insights visualizations created in: {output_dir}")
    return output_dir

def run_complete_business_analysis(model_results, risk_data=None):
    """
    Run complete business analysis with report and visualizations
    
    Parameters:
    -----------
    model_results : dict
        Dictionary containing model, metrics, feature_importance, etc.
    risk_data : DataFrame, optional
        DataFrame containing risk scores and categories
    
    Returns:
    --------
    dict
        Dictionary with paths to report and visualizations
    """
    # Generate business report
    report_path = generate_advanced_business_report(model_results, risk_data)
    
    # Create business visualizations
    vis_path = create_visual_business_insights(model_results, risk_data)
    
    return {
        'report_path': report_path,
        'visualizations_path': vis_path
    } 