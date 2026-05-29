# agents_assignment3_yaron_winter
Assignment 3 repo of the "From AI model to AI Product" class

# Repo Setup:
- Unzip agents_assignment3_yaron_winter.zip in your desired folder
- cd to the repo root (the folder generated following the unzip)
- Create conda environment with python>=3.11: # conda create -n <your conda env> python=3.11
- Activate the conda environment:             # conda activate your conda env
- Install required packages:                  # pip install -r requirements.txt
- Set the api key:                            # export NEBIUS_API_KEY=your api key

# Running the CLI application:
- python app.py                               # no checkpoint memory, default user profile
- python app.py --user name                   # no checkpoint memory, a specified user profile
- python app.py --session sname               # Use checkpoint memory (named sname), default user profile
- python app.py --session sname --user uname  # Use checkpoint memory (sname), a specified user profile (uname)

# MCP Server (FastMCP)
- Run the MCP Server: python mcp_client.py
- The MCP Server process:
  - The mcp client loads up the server
  - The server loads and stores the BiText table and defines the tools
  - The client displays (terminal, CLI) the tools provided by the server, with a brief explanation
  - We can get deeper info about each tool by using help (e.g. # help count_category)
  - Some examples are given below:

mcp>list_categories                        # list all categories of BiText

mcp>count_intent {"intent":"get_refund"}   # count the number of requests with get-refund intent

mcp>get_distribution_of_intents_for_category {"category":"PAYMENT"} # the intent distribution for the payment category

mcp> help count_intent                      # display the usage syntax of count_intent tool

# Architecture:
- All the agents implementation code is benath the agents folder:
  - dataset.py      # store the BiText table along a set of functions, which enables access to its data
  - display.py      # used for display each reasoning step
  - graph.py        # defintion and initialization of the agentic graph
  - profile.py      # manage the access to and the update of the user md profile file
  - react_nodes.py  # definition and initialization of the three react nodes (structured, unstructured, and personal)
  - router.py       # the router definition
  - type_tools.py # the tools of the structured and unstructured agents

# MCP Server:
- mcp_server.py   # definition of the mcp server
- mcp_client.py   # definition of the client
- mcp_confic.json # example for the configuration needed for connecting between the client and the server (not relevant for this local deployment)   
- Comments:
  - a personal agent was added to the graph, and the router was modified accordingly
  - when called, the personal agent considers ONLY the information given in user profile file
  - the router miss sometimes out-of-scope questions, and routes them to a reactor
  - but the react agent detect out-of-scope requests, and decline them gracefully
  - the memory / follow up questions work only partially (i.e. not always a follow up question is reconized and handled well), but it was beyond of my resources to try improve this...

# Model Choice:
- I chose the 'nvidia/nemotron-3..' models for this task
  - they were trained for instruct and reasoning, thus good for our purpose
  - their tokens' price is relatively low, comparing to other models, which is also good
- For the router I used the smaller one (nvidia/Nemotron-3-Nano-Omni), as the task of the router is pretty straightforward
- For the react agents I used the gigger one (nvidia/nemotron-3-super-120b-a12b), as they are required to perform much deeper analysis
