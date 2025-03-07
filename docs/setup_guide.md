# **Mac Python Environment Setup & Jupyter Lab Guide**

## **1. Check Python Installation**
First, check if Python is installed.

```bash
python3 --version
```

It is recommended to use Python 3.8 or later.

---

## **2. Create a Virtual Environment (`venv`)**
Navigate to your project directory and create a virtual environment.

```bash
python3 -m venv venv
```

Activate the created virtual environment (`venv/`).

```bash
source venv/bin/activate
```

When the virtual environment is active, the terminal prompt will show `(venv)`.

---

## **3. Install Required Libraries**
Use `requirements.txt` to install all necessary packages at once.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## **4. Set Up Jupyter Lab**
To run Jupyter Lab within the virtual environment, register the kernel.

```bash
python -m ipykernel install --user --name=venv --display-name "Python (venv)"
```

---

## **5. Start Jupyter Lab (Open in Browser)**
With the virtual environment activated, run the following command to open Jupyter Lab in a browser.

```bash
jupyter lab
```

---

## **6. Stop Jupyter Lab**
To stop Jupyter Lab, **press `Ctrl + C` in the terminal**.

---

## **7. Deactivate the Virtual Environment**
When you're done, deactivate the virtual environment.

```bash
deactivate
```

To reactivate the virtual environment next time, run:

```bash
source venv/bin/activate
```