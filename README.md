# applyai_tests
testing ground for all my theories on applied ai 

templates - 
create a template that can be reused for ongoing projects 




## Table of content

### Tests 

- [ ] clever-deductions: set up tool use [/tooluse]
- _composio_ : composio.io seems to be providing auth. 

### Infrastructure choices

- [ ] notes on operation

UX choice - 
I would like to be able to take gui projects and host them on the same domain url/(route:repo) 
for example - 
  combining (rowboat)[https://github.com/rowboatlabs/rowboat] and (docetl)[https://github.com/ucbepic/docetl]
  with a shared filesystem and interopterbility using a orchestrating agent - presents a paradigm of _agent controlled user defined workflows_ 
  - could this be done automatically, what architectural and infra decsions should i make?
      - _we could use screen use agents_ : start vm, download pakages, playwrite automation or nlweb or etc,


### Docs

- [ ] notes on documentation
- [ ] set up living documentaion 


keywords -
tool use 
  clever deductions 

task decomposition 

computer use- 
https://www.hud.so/
https://www.trycua.com/
https://developers.cloudflare.com/changelog/2025-05-28-playwright-mcp/

# Pages 

## Clever Deductions

create a mcp chat interface 
the interface should include the functionality of;
- A "Hub" - UX of urls - a user can oneclick install a mcp server from a list, install there own tool to a mcp server, and recieve user.tooluse.com/{user:tool_key}/sse
