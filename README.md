# Information Retrieval and Web Analytics (IRWA) - G23 Final Project

<table>
  <tr>
    <td style="vertical-align: top;">
      <img src="static/image.png" alt="Project Logo"/>
    </td>
    <td style="vertical-align: top;">
      This repository contains the code for the IRWA Final Project - Search Engine with Web Analytics.
      The project is implemented using Python and the Flask web framework. It includes a simple web application that allows users to search through a collection of documents and view analytics about their searches.
    </td>
  </tr>
</table>

----
## Project Structure

```
/irwa-search-engine
├── myapp                # Contains the main application logic
├── templates            # Contains HTML templates for the Flask application
├── static               # Contains static assets (images, CSS, JavaScript)
├── data                 # Contains the dataset file (fashion_products_dataset.json)
├── project_progress     # Contains your solutions for Parts 1, 2, and 3 of the project
├── .env                 # Environment variables for configuration (e.g., API keys)
├── .gitignore           # Specifies files and directories to be ignored by Git
├── LICENSE              # License information for the project
├── requirements.txt     # Lists Python package dependencies
├── web_app.py           # Main Flask application
└── README.md            # Project documentation and usage instructions
```


----
## To download this repo locally

Open a terminal console and execute:
```
cd <your preferred projects root directory>
git clone https://github.com/LucaFranceschi01/IRWA-2025-G23.git
```

## Setting up the Python environment (only for the first time you run the project)
### Install virtualenv
Setting up a virtualenv is recommended to isolate the project dependencies from other Python projects on your machine.
It allows you to manage packages on a per-project basis, avoiding potential conflicts between different projects.

In the project root directory execute:
```bash
pip3 install virtualenv
virtualenv --version
```

### Prepare virtualenv for the project
In the root of the project folder run to create a virtualenv named `irwa_venv`:
```bash
virtualenv irwa_venv
```

If you list the contents of the project root directory, you will see that it has created a new folder named `irwa_venv` that contains the virtualenv:
```bash
ls -l
```

The next step is to activate your new virtualenv for the project:
```bash
source irwa_venv/bin/activate
```

or for Windows...
```cmd
irwa_venv\Scripts\activate.bat
```

This will load the python virtualenv for the project.

### Installing Flask and other packages in your virtualenv
Make sure you are in the root of the project folder and that your virtualenv is activated (you should see `(irwa_venv)` in your terminal prompt).
And then install all the packages listed in `requirements.txt` with:
```bash
pip install -r requirements.txt
```

If you need to add more packages in the future, you can install them with pip and then update `requirements.txt` with:
```bash
pip freeze > requirements.txt
```

Enjoy!

## Load the data

Put the data file `fashion_products_dataset.json` and `validation_labels.csv` in the `data` folder. It was provided to us by the instructor.

## Starting the Web App
```bash
python -V
# Make sure we use Python 3

cd search-engine-web-app
python web_app.py
```
The above will start a web server with the application:
```
 * Serving Flask app 'web-app' (lazy loading)
 * Environment: production
   WARNING: This is a development server. Do not use it in a production deployment.
   Use a production WSGI server instead.
 * Debug mode: off
 * Running on http://127.0.0.1:8088/ (Press CTRL+C to quit)
```

Open Web app in your Browser:  
[http://127.0.0.1:8088/](http://127.0.0.1:8088/) or [http://localhost:8088/](http://localhost:8088/)

## Tunnelling the Web App to get location information

Since if running on localhost there's no possibility of tracking location from IP Address, we have tunnelled our localhost application through Cloudflare's and accessed it through a public generated URI. After that, we've parsed the IP and queried `IpInfo.io` for information about location. We've stored the result and used it in the dashboard.

The steps to get the tunnelling working are fairly easy:

1. Install `cloudflared` [Download link for all OSs](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/downloads/#latest-release)
2. Once the WebApp is up and running in localhost, tunnel that port using the command
```
cloudflared tunnel --url http://127.0.0.1:8088
```
3. The above command will eventually show a URL from which the application can be accessed. Note: the location will be recorded in the dashboard!
```
...
Requesting new quick Tunnel on trycloudflare.com...
2025-11-29T12:17:35Z INF +--------------------------------------------------------------------------------------------+
2025-11-29T12:17:35Z INF |  Your quick Tunnel has been created! Visit it at (it may take some time to be reachable):  |
2025-11-29T12:17:35Z INF |  https://this-would-be-the-url.trycloudflare.com                                           |
2025-11-29T12:17:35Z INF +--------------------------------------------------------------------------------------------+
...
```

> If there's any problem replicating these steps, we are more than willing to set this up for evaluation so the teacher just has to enter a link. Just reach out!

## Usage: 
1. As for Parts 1, 2, and 3 of the project, please use the `project_progress` folder to store your solutions. Each part should contain `.pdf` file with your report and `.ipynb` (Jupyter Notebook) file with your code for solution and `README.md` with explanation of the content and instructions for results reproduction.
2. For the Part 4, of the project, you should build a web application using Flask that allows users to search through a collection of documents and view analytics about their searches. You should work mailnly in the `web_app.py` file `myapp` and `templates` folders. Feel free to change any code or add new files as needed. The provided code is just a starting point to help you get started quickly.
3. Make sure to update the `.env` file with your Groq API key (can be found [here](https://groq.com/), the free version is more than enough for our purposes) and any other necessary configurations. IMPORTANT: Do not share your `.env` file publicly as it contains sensitive information. It is included in `.gitignore` to prevent accidental commits. (It should never be included in the repos and appear here only for demonstration purposes).
4. Have fun and be creative!

## Attribution:
The project is adapted from the following sources:
- [IRWA Template 2021](https://github.com/irwa-labs/search-engine-web-app)
- [IRWA Template 2025](https://github.com/trokhymovych/irwa-search-engine)
