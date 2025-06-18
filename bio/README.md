# Biogas Production Prediction Project

A machine learning project to predict biogas yield from anaerobic digestion using livestock data.

## Project Overview
This project aims to predict biogas generation from different types of livestock waste using machine learning techniques. The model takes into account various parameters such as:
- Animal counts (Cattle, Dairy, Poultry, Swine)
- Digester Type
- Year of Operation
- Co-Digestion status

## Project Structure
```
biogas_prediction/
├── bio/
│   ├── data/
│   │   └── __init__.py
│   ├── models/
│   │   └── __init__.py
│   ├── preprocessing/
│   │   └── __init__.py
│   ├── visualization/
│   │   └── __init__.py
│   ├── notebooks/
│   │   └── exploratory.ipynb
│   └── main.py
├── requirements.txt
└── README.md
```

## Installation
1. Clone the repository
2. Create a virtual environment:
```bash
python -m venv biogas_env
```
3. Activate the virtual environment:
```bash
# Windows
biogas_env\Scripts\activate
```
4. Install dependencies:
```bash
pip install -r requirements.txt
```

## Model Performance
- RMSE: 0.72
- R² Score: 0.57

## Dataset
The project uses the AgSTAR Livestock Anaerobic Digester Database, which includes:
- 429 entries
- 23 features
- Target variable: Biogas Generation Estimate (cu-ft/day)

## Technologies Used
- Python 3.x
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

## Contributing
Feel free to fork the project and submit pull requests.

## License
MIT License