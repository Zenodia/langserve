# LC Action Agents

Implementation of Action Agents on top of the LangChain / LangGraph framework.

We implements 2 agents:
- Flight simulator Agent
- VRED Car configuerator agent

## Flight Simulator Agent
The flight simulator 3d app that we are integrating with could be downloaded from here: https://www.x-plane.com/ (it includes free demo version).

The code of the agent is located in: source>agents>xplane folder of the project. Flight simulator app includes rich set of commands which could be triggered via keyboard bindings. We simulate triggering of these keyboard bindings from the AI Agent workflow to trigger the requests commands in the app.

You must have the x-plane simulator app up and running before spin up the server.

## Installing and running

Install Python packages from the requirements.txt

```bash
pip install -r requirements.txt
```


## Spin up langchain server 
cd into nv3dagent and run the below command to spin up the server

```shell
poetry run langchain serve --port=8100
```
## go to the client jupyter notebook 