# knowledge-query-portal

The Knowledge Query Portal is a Flask-based web application that enables employees of an organization to upload, process, and interact with documents using natural language. The application supports both unstructured PDF documents and structured tabular data (Excel/CSV). It leverages LangChain to allow users to "chat" with documents—using LLMs to answer questions about document content or to generate SQL queries for tabular data. Role-based access control ensures that users see only the documents they are authorized to access.

Below are the features of the application:
(i) All employees can access the existing functionalities of chatting with documents i.e. Upload documents, Select Documents, Ask, View Files

(ii) Each employee can have only a type of access_level among - project, department, admin
- An employee with admin role (access_level) has access to all documents uploaded by any employee, 
- An employee with department role (access_level) has access to all those documents uploaded by employees of all projects mapped to that specific department.
- An employee with project role has access (access_level) to those documents uploaded by employees of mapped to the specific project."

(iii) Each project is tagged to a department. Multiple projects can be mapped to a single department but a single project cannot be mapped to multiple departments. 

(iv) An admin can register employee, delete employee apart from the other functionality which are available for all employee

(v) Multiple PDF documents can be selected and processed which forms base knowledge for the chatbot.

(vi) Incase of Tabular data multiple excel/csv files can be selected and processed, while doing that prompt can entered to mention the common key between the files.  


![flowchart](https://github.com/user-attachments/assets/cb204cb8-bea8-4b0c-b0b3-fcd2b6fec72f)
