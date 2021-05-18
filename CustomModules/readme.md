Author: Rodrigo Doering Neves

Email: rodrigodoeringneves@gmail.com

Phone: +55 11 98503-1110 (São Paulo - Brasil)

- This repository contains custom modules and packages written for a variety of purposes, both professional and personal.
- Since no sensible information exists within this modules, I decided to upload it as a public repos to facilitate importing the modules to other environments but also to easily share with others
- Each folder here represents either a custom package with sub-modules or a single module

Description of Modules:

1) Database : module for integration between relational databases and Python. Primarly written to integrate with Microsoft SQL Server and Microsoft Azure SQL via ODBC Driver connection implemented with pyodbc module. Supports several relational database taskes such as: 
       
       - Creating databases
       - Creating and accessing tables and schemas
       - Write and Read data from specific SQL Server databases
       - Update data
       - Import and export tables and full databases as .csv, .xlsx and .json formats
       - For now it only supports Microsoft databases as mentioned


2) MathPack : module related to mathematical operations and demonstrations, also for didactic purposes:
       
       - MathPack.functions module contains plain functions for computing specific equations and techniques
       - MathPack.display module uses LaTex code to display matrices and vectors in a nice fashion, mostly used in Jupyter Notebooks



3) Visuals: package for automatization of Matplotlib.pyplot based visualizations written as classes (OOP):
       
       - Visuals.plots contains Plot object, the main class to be imported on presentations and jupyter notebooks
       - For now, supports ploting functions, scatter plots, surfaces (both 2D and 3D) and vectors with a variety of possible customizations
       - It allows multiple calls and graphs in the same plot for complex visualizations (which is basically the main goal of this module - writting nice plots with few lines)
       - Visuals._base_ and Visuals._utils_ are subpackages containing the classes and functions imported to Visuals.plots.Plot
       - Visuals._base_._graph_base.GraphBase_ is the Superclass inherited by Plot class
       - Visuals._base_._axes_base.AxesInstance_ has a composition relation with Visuals._base_._graph_base.GraphBase_
 

