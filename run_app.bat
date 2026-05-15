@echo off
echo Activating FYP Environment...
call conda activate FYP

echo Starting Streamlit...
cd "C:\Users\Mr.Khan\Desktop\MLOps-class_task"
python -m streamlit run app.py

pause