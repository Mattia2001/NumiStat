# NumiStat
Future project involving deep learning to determine coins conservation state and predict their auction price. 

# Deep Learning for Numismatics

A research-oriented software project for extracting structured data from historical auction catalogs and building AI systems for numismatic analysis.

## Vision

This project aims to build an end-to-end AI pipeline for numismatics, starting from public auction documents and evolving toward decision-support tools for collectors, professionals, and auction houses.

The project focuses initially on single coins and uses auction catalogs and realized-price documents to create a structured dataset suitable for:

- statistical analysis of the numismatic market
- image-based coin classification
- assisted estimation of the state of conservation
- realized price modeling and price prediction
- future conversational and agentic analytics over the market dataset

The initial data source is the online documentation of the auction house Nomisma. The architecture is designed so that it can later be extended to other auction houses such as Baldwin's.

## Project Goals

The project is organized around four progressive goals.

### 1. Build a high-quality dataset

Auction catalogs contain a large amount of valuable information, but it is embedded in semi-structured text and images rather than clean tables. The first objective is therefore to transform PDF catalogs and realized-price documents into a structured dataset.

The dataset must preserve both:

- the original raw text for each lot
- normalized structured fields extracted from it

The initial scope is intentionally restricted to:

- single coins only
- no coin sets
- no banknotes
- no watches
- no fountain pens
- no other collectibles

This makes the first version of the pipeline more coherent and easier to validate.

### 2. Automate feature extraction from auction documents

The second objective is to automate the extraction of relevant features from auction documents.

The extraction pipeline is expected to combine:

- deterministic parsing for highly regular fields
- optional LLM-based semantic extraction for ambiguous or descriptive fields

This hybrid strategy is preferred over a purely deterministic parser, because catalog descriptions are semi-structured and often include expert commentary, qualitative judgments, damage notes, rarity remarks, slab information, and provenance details in variable order.

### 3. Build AI models on top of the dataset

Once a sufficiently large structured dataset is available, the next objective is to train machine learning and deep learning models for two main tasks:

- coin image classification and assisted conservation grading
- price estimation / price prediction

These two tasks have different input modalities and may require different model families.

### 4. Evolve into an interactive web and agentic analytics platform

In a later stage, the project may evolve into a web application where users can upload images, browse historical data, and interact with a conversational system capable of querying and analyzing the numismatic market dataset.

The long-term idea is not only to build predictive models, but also a market intelligence tool for numismatics.

## High-Level Architecture

The project is organized as a pipeline with modular stages.

```text
Raw auction documents
    ↓
Document ingestion
    ↓
Lot segmentation
    ↓
Deterministic parsing + optional LLM extraction
    ↓
Structured dataset
    ↓
Data validation and normalization
    ↓
Model training
    ├─ Image classification / conservation estimation
    └─ Price modeling / regression
    ↓
Evaluation
    ↓
Web application / conversational analytics / agentic layer
```

## Core Architectural Blocks

### 1. Dataset Construction Layer

This is the foundational block of the project.

Its role is to ingest auction documents and create a trustworthy dataset. Without this layer, all downstream ML tasks become weak or misleading.

#### Inputs

- auction catalog PDFs
- realized-price PDFs
- possibly HTML pages or structured auction pages in future versions
- lot images extracted from PDFs or collected from online sources

#### Outputs

A structured dataset where each row corresponds to a single coin lot and includes:

- auction metadata
- lot metadata
- parsed textual attributes
- normalized numismatic fields
- price information
- references to associated images
- original raw text for traceability

#### Design principle

The raw source text must always be preserved. Structured fields are derived views, not replacements.

This is important because:

- parsing rules may improve over time
- LLM extraction prompts may evolve
- some fields may require re-interpretation later
- the original expert wording does itself contain market value

### 2. Parsing and Feature Extraction Layer

This is the first major technical node of the project.

Auction descriptions are not clean tables. They are semi-structured expert text. A robust extraction strategy must therefore distinguish between:

- strong fields, which can often be extracted deterministically
- weak fields, which may require semantic interpretation

#### Strong fields

These are usually good candidates for regex-based or rule-based extraction:

- lot number
- catalog reference
- metal
- year
- weight
- diameter
- noble metal percentage
- raw rarity notation
- raw conservation notation
- slab company
- slab grade
- slab certificate number
- base price
- hammer price

#### Weak fields

These may be better handled by an LLM or by later semantic normalization:

- issuer or ruler
- issuing authority / colony / region
- denomination when phrasing varies
- positive qualitative notes
- defect descriptions
- patina notes
- cleaning / mounting / scratches / corrosion remarks
- provenance commentary
- market commentary embedded in the description

#### Recommended extraction strategy

A two-stage architecture is recommended:

1. deterministic parser  
   Segment the document into lots and extract highly regular fields.

2. semantic extraction layer  
   Use an LLM, when useful, to interpret ambiguous text and output normalized JSON for each lot.

This keeps the system:

- auditable
- reproducible
- easier to debug
- more robust than a fully free-form LLM pipeline

### 3. Validation and Gold Dataset Layer

Before scaling the pipeline, a manually curated gold reference is required.

The current plan is to build a gold sample of 100–200 single-coin lots from Nomisma Auction 71.

This gold dataset will serve several purposes:

- evaluate parser performance
- evaluate LLM extraction quality
- identify ambiguous fields
- define normalization rules
- create a reliable benchmark before scaling up

#### Suggested fields for the gold sample

- `lot_number`
- `raw_text`
- `issuer`
- `denomination`
- `year`
- `metal`
- `rarity_raw`
- `grade_raw`
- `reference_raw`
- `defect_notes_raw`
- `positive_notes_raw`
- `base_price`
- `hammer_price` if available
- `notes_for_parser` for ambiguous or difficult cases

## Machine Learning Roadmap

### Task A: Image-Based Coin Classification and Conservation Support

This branch focuses on visual analysis of the coin.

#### Objective

Train a DNN to classify coins from images and assist in estimating the state of conservation.

The aim is not to replace the numismatist, but to support expert assessment in the same general spirit in which AI assists human specialists in medical imaging: the final judgment remains human, but the model may help with consistency, screening, and first-pass evaluation.

#### Possible sub-tasks

- obverse/reverse type classification
- ruler / denomination classification
- metal or broad series classification
- conservation grade estimation
- ordinal classification of conservation levels
- detection of defects or anomalies

#### Notes

Conservation grading is inherently difficult because it depends on:

- lighting conditions
- image quality
- wear patterns
- patina
- reflections
- cleaning traces
- scratches or edge damage
- expert subjectivity

For this reason, the model should initially be framed as an assistive tool, not an automatic authority.

### Task B: Price Modeling and Prediction

This branch focuses on market behavior.

#### Objective

Estimate the expected hammer price or a plausible realized price range for a coin lot.

#### Important modeling note

This is not simply a time-series forecasting problem.

In most cases it is better framed as:

- tabular regression
- multimodal regression
- ranking / market scoring
- uncertainty-aware price estimation

Potential inputs include:

- structured numismatic attributes
- conservation information
- auction house identity
- auction date
- textual lot description
- image-derived features
- external market variables

#### Why model selection matters

Some very rare coins may appear only a few times in the dataset. This means the data can be sparse, highly imbalanced, and heterogeneous.

A naive regression model may therefore fail badly on rare or exceptional lots.

Modeling strategies may include:

- strong baselines with gradient boosting on tabular data
- multimodal models combining tabular, text, and image features
- uncertainty-aware prediction
- quantile regression for price intervals
- similarity-based retrieval using comparable historical lots
- hierarchical models by category, ruler, denomination, or auction house

The goal is not to produce a magical exact price, but a decision-support estimate with context and uncertainty.

## Why This Project Can Produce Value

The project is intended to support market understanding rather than speculative automation.

### For buyers and collectors

It may help answer questions such as:

- Is the auction base reasonable?
- How did similar coins perform historically?
- Which auction house tends to realize higher prices for this category?
- How stable or volatile is the market for this type?
- What are the main factors associated with higher realizations?

### For auction houses

It may support:

- more informed pricing
- analytics over historical realized prices
- identification of strong or weak market segments
- improved consistency in cataloging and valuation support

### For researchers and professionals

It provides a structured historical dataset for:

- statistical analysis
- market studies
- trend analysis
- comparative analysis between auction houses
- testing of multimodal ML methods on a specialized domain

## Possible Web Application Evolution

A future web application could expose the project through several modules.

### Module 1: Document-to-dataset ingestion

Upload catalog documents and automatically extract structured lots.

Possible functions:

- PDF upload
- parsing preview
- human validation interface
- export to CSV / JSON / database

### Module 2: Coin image analysis

Upload one or more images of a coin and receive:

- predicted class or family
- suggested denomination / ruler / type
- estimated conservation range
- warnings about uncertainty or low confidence

### Module 3: Price analysis and forecasting

Given a coin profile, the system could provide:

- estimated hammer price range
- comparable historical lots
- auction-house-aware price analysis
- uncertainty indicators
- visual explanation of relevant variables

### Module 4: Market intelligence dashboard

A market analytics dashboard could include:

- realized price trends over time
- comparisons across auction houses
- heatmaps by ruler / denomination / mint / period
- conservation-price relationships
- rarity-price relationships
- liquidity indicators by category
- descriptive statistics and correlations

## Agentic AI and Conversational Layer

A later evolution of the project could introduce an agentic AI layer.

The idea is to allow the user to interact with the dataset through a chatbot or analyst-style assistant capable of exploring the data and returning structured answers, summaries, and insights.

### Example use cases

The user may ask questions such as:

- Show the historical realized prices for 20 Lire of Umberto I across different auction houses.
- Which Nomisma coin categories have shown the strongest growth over the last five years?
- How does conservation affect realized prices for a given denomination?
- Are there correlations between precious metal prices and realized auction prices?
- Which auction house has historically performed best for a certain segment?
- Which categories appear most underpriced relative to comparable historical lots?

### Potential external signals

In future versions, the system could correlate auction data with external variables such as:

- precious metal prices
- inflation indicators
- interest rates
- macroeconomic indicators
- broad collectible market sentiment proxies

### Value of the agentic layer

This would turn the project from a static ML pipeline into an interactive market intelligence platform.

Such a system could help:

- buyers identify opportunities
- professionals analyze comparables more efficiently
- auction houses understand demand and pricing dynamics
- researchers query a specialized market dataset without writing code

## Suggested Repository Structure

```text
deep-learning-numismatics/
│
├─ README.md
├─ data/
│  ├─ raw/
│  ├─ interim/
│  ├─ processed/
│  └─ gold/
│
├─ docs/
│  ├─ architecture/
│  ├─ prompts/
│  └─ datasets/
│
├─ notebooks/
│  ├─ 01_gold_sample_exploration.ipynb
│  ├─ 02_parser_baseline.ipynb
│  ├─ 03_llm_extraction.ipynb
│  ├─ 04_image_model_baseline.ipynb
│  └─ 05_price_model_baseline.ipynb
│
├─ src/
│  ├─ ingestion/
│  ├─ parsing/
│  ├─ extraction/
│  ├─ normalization/
│  ├─ validation/
│  ├─ vision/
│  ├─ pricing/
│  ├─ analytics/
│  └─ app/
│
├─ tests/
└─ configs/
```

## Immediate Next Steps

1. Complete the gold reference dataset for 100–200 single-coin lots from Nomisma Auction 71.
2. Define the target JSON schema for a parsed lot.
3. Implement a deterministic baseline parser for strong fields.
4. Add an optional LLM-based semantic extraction stage for weak fields.
5. Evaluate extraction quality against the gold dataset.
6. Scale the ingestion pipeline to a larger set of Nomisma auctions.
7. Start baseline experiments for image modeling and price modeling.

## Guiding Principles

- Preserve the original raw lot text.
- Keep extraction auditable and reproducible.
- Use deterministic parsing where possible.
- Use LLMs selectively for semantic interpretation.
- Treat conservation grading as expert support, not expert replacement.
- Treat price prediction as decision support with uncertainty, not as a guaranteed valuation.
- Design the system so it can scale from one auction house to many.

## Status

Early-stage research and architecture definition.

The current focus is:

- gold dataset preparation
- extraction architecture design
- parser + optional LLM strategy
- planning the ML roadmap for vision and pricing
