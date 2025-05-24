# Development of a Machine Learning-Based Prototype for Cinema Ticket Sales Forecasting

This project is built as part of a bachelor's thesis 
and is a .NET 8.0–based prototype 
that leverages ML.NET to predict per‐showtime ticket sales and 
exposes these forecasts via an ASP.NET Core REST API. 

## Background

**Apollo Kino** is the largest cinema chain in the Baltics, operating 19 multiplex locations (13 in Estonia, 3 in Latvia, 3 in Lithuania). 
Each multiplex contains multiple auditoriums of varying capacities and seating types.

The dataset comprises **over 155,000** historical sessions:
- Time period: 2022 July - 2024 July
- Minimum tickets sold: 1  
- Maximum tickets sold: 918  
- Sessions with ≤ 50 tickets sold: 86.5% of data  
- Sessions with ≤ 150 tickets sold: 99.1% of data 

**Data features** span session metadata (cinema, city, country, date/time), movie attributes (title, genre, rating, length, language, weeks since release), and performance indicators (e.g. average and recent ticket sales across different time periods).  
The prediction **label** for each session is its ticket sales count.

The data is inherently **time-series** - care was taken to avoid data leakage by preserving temporal order during data preparation, model training, and validation.  
Historical data was split chronologically to ensure that training data precedes test data, reflecting realistic deployment conditions and maintaining the integrity of performance estimates.


## Overview

The goal of this project was to replace intuition-based showtime scheduling with a data-driven forecasting tool for Apollo Kino’s cinema chain.  
The project entails data preparation, machine learning model development, evaluation, and API development to forecast cinema ticket sales at the showtime level.

### Data Preparation  
- Based on session‐level historical ticket sales and film metadata from Apollo Kino’s database.  
- Feature engineering.  
- Models tested on raw and logarithmically transformed ticket counts due to heavy right-skew in the data.

### Model Development & Validation  
- Compared all regression algorithms available in ML.NET.
- Selected FastTree as the best-performing algorithm.
- Conducted a hyperparameter grid search to tune the model.  
- Optimized the model's features to achieve the best-performing model.

### Results  
- Mean Absolute Error on hold-out data: ~10 tickets per session.  
- Coefficient of determination (R²): ~0.5, indicating a moderate to strong relationship between features and ticket sales.

### API Deployment  
- Serialized the trained model into a `.zip` artifact.  
- Implemented an ASP.NET Core Web API exposing endpoints to:  
  - Retrieve a single-session forecast  
  - Retrieve multi-session forecasts  
  - Generate time-series forecasts for different show times  
  - Identify best session times within a time range  

This end-to-end .NET prototype demonstrates how ML.NET and ASP.NET Core can be combined to deliver a data-driven forecasting tool for cinema scheduling.  

## Solution Folder Level Structure
The solution is organized into three main solution folders:

- **0_solution_items**  
  Configuration files, `.gitignore`, `Directory.Build.props`, global `README.md`.
- **1_app**  
  Core application projects and analysis tools.
- **2_base**  
  Utility classes.

## Core Project Structure
The following describes the structure and intended purpose of the projects inside the 1_App solution folder._

- **App.DTO/**  
  Defines the request/response data transfer objects (DTOs) for the API.

- **ML.Analysis/**  
  Reproduces all thesis‐level analyses (general model evaluations, hyperparameter grid search, PFI, rolling-origin CV) and exports results to CSV.

- **ML.Data/**  
  Implementations of `ITrainingDataLoader` for loading training data from files (intended for development) or a database (intended for production).

- **ML.Domain/**  
  Defines input schemas and feature mappings for the ML model’s data along with final model feature definitions.

- **ML.Model/**  
  Classes for building the ML model pipeline and the definition of the developed FastTree model.

- **ML.Services/**  
  - **InferenceEngine**: runs predictions from the saved `.zip` model  
  - **TrainerService**: loads data, trains & serializes the model  
  - **TrainingScheduler**: triggers periodic retraining (e.g. weekly)

- **WebApp/**  
  ASP.NET Core application exposing five REST endpoints for single/multi-session forecasts with DI configuration.

## Future Development Ideas

The current prototype uses only Apollo Kino’s internal session-level ticket sales data, offering a transparent, reproducible, and fully C#/.NET-based alternative to closed-source corporate forecasting tools.  

Beyond this scope, several enhancements could further improve accuracy and usability:

- **Incorporate External Data for Cold-Start Films**  
  New releases lack historical sales, which weakens accuracy. Integrating movie ratings (IMDb, Rotten Tomatoes) or cumulative box-office revenue can help predict demand for brand-new titles.

- **Address High-Sales Outliers**  
  Sessions with exceptionally high attendance are underrepresented in the dataset. External indicators of the movie's general performance so far could improve model's accuracy on these tail cases.

- **Enrich Feature Set with Contextual Factors**  		
  Factors like weather forecasts, competing showtimes at nearby cinemas, holiday schedules, and local events (concerts, sports) can be incorporated to capture demand fluctuations more accurately.

- **Automatic Full-Day Scheduling**  
  Extend the API to not only forecast individual sessions but to propose a full-day schedule that maximizes predicted ticket sales across all auditoriums in the cinema.

- **Scheduling Optimization Integration**  
  Combine the forecasting prototype with an optimization algorithm (e.g., integer programming) to generate resource-efficient, revenue-maximizing showtime plans.

These improvements would build on the existing transparent, reproducible foundation and move toward a fully automated, end-to-end cinema scheduling solution.  
