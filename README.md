# GPT From Scratch (PyTorch)

A minimal , interpretable implementation of GPT from scratch using PyTorch.

## Overview 

This repository contains code-implementation of a GPT-style Language Model from the ground-breaking research paper "Attention is All you Need":

The goal is to demonstrate deep understanding of: 
- Tokenization and Vocabulary Creation
- Token Embeddings and Position Embeddings
- Causal (Masked) Self-Attention
- Transformer Blocks 
- Training the Model to Predict the Next token
- Autoregressive Text Generation

All of these are implemented in the notebook without using high-level Transformer libraries.

## What this project Demonstrates 
- Word-level Tokenization
- Encoding text into a continous token stream
- Causal Langual Modelling by shifting targets
- Masked Self Attention (Single Head and Multi Head)
- Feed Forward Neural Network for token wise computation
- Residual Connection and Layer Normalization
- Combining (MSA + FFN + Residual Connection + LayerNorm + Vocab Projection) to form GPT
- Training Loop
- Autoregressive Generation with strict `<END>` stopping
