# agents_assignment3_yaron_winter
Assignment 3 repo of the "From AI model to AI Product" class

# Repo Setup:
- Unzip agents_assignment3_yaron_winter.zip in your desired folder
- CD to the repo root (the folder generated following the unzip)
- Create conda environment with python>=3.11: # conda create -n <your conda env> python=3.11
- Install required packages:                  # pip install -r requirements.txt
- Set the api key:                            # export NEBIUS_API_KEY=your api key
- Activate the conda environment:             # conda activate your conda env

# Running the CLI application:
- python app.py                               # no checkpoint memory, default user profile
- python app.py --user name                   # no checkpoint memory, a specified user profile
- python app.py --session sname               # Use checkpoint memory (named sname), default user profile
- python app.py --session sname --user uname  # Use checkpoint memory (sname), a specified user profile (uname)

## MCP server (FastMCP)
- Run the MCP Server: python mcp_client.py

mcp> list                                   # re-list available tools
mcp> help count_category                    # show a tool's input schema
mcp> list_categories                        # call a no-arg tool
mcp> count_category {"category": "REFUND"}  # call with JSON arguments
mcp> count_category category=REFUND         # ...or key=value shorthand
mcp> exit
