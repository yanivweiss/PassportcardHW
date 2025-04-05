import json
import os

# Create notebooks directory if it doesn't exist
os.makedirs('notebooks', exist_ok=True)

# Template for PassportCard_Insurance_Claims_Prediction.ipynb
claims_notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# PassportCard Insurance Claims Prediction\n\nThis notebook develops a machine learning system to predict future insurance claims for PassportCard policyholders."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(\"Welcome to the Insurance Claims Prediction notebook!\")"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Template for PassportCard_Model_Development.ipynb
model_notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# PassportCard Model Development\n\nThis notebook focuses on developing and evaluating machine learning models for insurance claims prediction."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(\"Welcome to the Model Development notebook!\")"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Template for PassportCard_Business_Applications.ipynb
business_notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# PassportCard Business Applications\n\nThis notebook explores the business applications and insights from our insurance claims prediction models."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "print(\"Welcome to the Business Applications notebook!\")"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.8.10"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Save the notebooks to files
with open('notebooks/PassportCard_Insurance_Claims_Prediction.ipynb', 'w') as f:
    json.dump(claims_notebook, f, indent=1)

with open('notebooks/PassportCard_Model_Development.ipynb', 'w') as f:
    json.dump(model_notebook, f, indent=1)

with open('notebooks/PassportCard_Business_Applications.ipynb', 'w') as f:
    json.dump(business_notebook, f, indent=1)

print("Notebooks created successfully!") 