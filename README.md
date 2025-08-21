[flant5_legal_readme.md](https://github.com/user-attachments/files/21926638/flant5_legal_readme.md)
# FLAN-T5 Legal Question Answering with Supervised Fine-Tuning

A comprehensive study comparing different fine-tuning approaches for adapting FLAN-T5 to legal question answering tasks. This project demonstrates three distinct training methodologies and evaluates their effectiveness on legal domain questions.

## 🎯 Project Overview

This project fine-tunes Google's FLAN-T5-small model for legal question answering using multiple supervised fine-tuning (SFT) approaches:

1. **Baseline Model**: Original FLAN-T5-small without fine-tuning
2. **Span Corruption Model**: T5-style span corruption training on legal text
3. **Q&A Fine-tuned Model**: Direct supervised fine-tuning on legal Q&A pairs

## 📊 Key Results

This project was designed as a **learning exercise** to practice different fine-tuning approaches rather than to achieve production-ready results. The outcomes demonstrate important limitations:

- **Limited Base Capability**: FLAN-T5-small lacks fundamental legal knowledge for meaningful legal reasoning
- **Model Destabilization**: Fine-tuning on such a small, specialized dataset may destabilize the model's general capabilities
- **Insufficient Training Data**: 58 Q&A pairs and 211 span corruption examples are inadequate for domain transfer
- **Educational Value**: Successfully demonstrates fine-tuning mechanics and comparative evaluation methods

## 🏗️ Architecture & Methodology

### Model Configuration
- **Base Model**: `google/flan-t5-small` (77M parameters)
- **Framework**: Transformers + PyTorch
- **Training Environment**: Google Colab
- **Evaluation**: Qualitative comparison across legal domains

### Training Approaches

#### 1. Span Corruption Training
```python
# T5-style masked language modeling with legal text
training_examples = create_span_corruption_data(legal_docs, mask_probability=0.15)
```
- **Data**: 211 training examples from legal documents
- **Method**: Mask legal terms and predict missing spans
- **Epochs**: 3 epochs with early stopping

#### 2. Q&A Supervised Fine-tuning
```python
# Direct question-answer pair training
training_data = load_legal_qa_datasets()  # 58 Q&A pairs
```
- **Data Sources**: WikiQA + synthetic legal Q&A pairs
- **Method**: Direct sequence-to-sequence training
- **Epochs**: 5 epochs with validation monitoring
```

## 🔧 Installation & Setup

### Prerequisites
```bash
pip install torch transformers datasets requests beautifulsoup4
```

### Quick Start
1. Clone the repository
2. Open `flant5_legal_sft.ipynb` in Google Colab
3. Run all cells to reproduce the training pipeline
4. Models will be saved locally for inference

## 📚 Dataset Construction

### Legal Document Sources
- **Cornell Law School Wex**: Legal encyclopedia entries
- **Synthetic Legal Text**: General legal concepts and terminology

### Q&A Dataset Creation
- **WikiQA**: Legal-filtered questions from Wikipedia
- **Synthetic Pairs**: 28 high-quality legal Q&A pairs covering basic legal concepts

## 🧪 Experimental Framework

This project serves as a **practical exercise in fine-tuning techniques** rather than an attempt to create a production legal AI system. The experimental design focuses on:

### Learning Objectives
- Compare different fine-tuning methodologies (span corruption vs. direct Q&A)
- Practice dataset creation and preprocessing for domain-specific tasks
- Understand the challenges of adapting small language models to specialized domains
- Explore the trade-offs between different training approaches

### Known Limitations
- **Base Model Constraints**: FLAN-T5-small (77M parameters) lacks the capacity for complex legal reasoning
- **Data Insufficiency**: Limited training data cannot provide comprehensive legal knowledge
- **Domain Mismatch**: General-purpose models struggle with highly specialized legal terminology and concepts
- **Potential Degradation**: Small-scale domain fine-tuning may harm the model's general capabilities

### Test Questions (15 Legal Questions)
*Used to demonstrate fine-tuning effects rather than validate legal accuracy*

The evaluation uses 15 diverse legal questions spanning different areas of law to test the models' responses and compare the effects of different fine-tuning approaches.

## 📈 Key Findings & Lessons Learned

### Technical Insights
1. **Base Model Limitations**: FLAN-T5-small fundamentally lacks the knowledge base for legal reasoning
2. **Fine-tuning Challenges**: Domain-specific fine-tuning on small datasets can destabilize model performance
3. **Data Requirements**: Meaningful domain transfer requires substantially more training data than provided
4. **Training Methodology**: Successfully implemented and compared multiple fine-tuning approaches

### Important Limitations Discovered
- **Knowledge Gap**: The base model has insufficient legal knowledge to build upon
- **Model Degradation Risk**: Fine-tuning may harm the model's general language capabilities
- **Evaluation Challenges**: Comparing poor outputs doesn't validate the approach
- **Resource Constraints**: Small models are inadequate for complex domain reasoning
## 🔄 Training Pipeline

```python
# Complete pipeline execution
if __name__ == "__main__":
    # Step 1: Span corruption training
    model, tokenizer = fine_tune_t5_on_legal_mlm()
    
    # Step 2: Q&A model training  
    qa_model, qa_tokenizer = fine_tune_qa_model()
    
    # Step 3: Comprehensive evaluation
    all_responses = compare_all_three_models()
```
## 🔬 Lessons for Future Projects

### What Would Work Better
1. **Larger Base Models**: Use models with existing legal knowledge (7B+ parameters)
2. **Retrieval-Augmented Generation**: Combine small models with legal databases
3. **Massive Datasets**: Thousands of high-quality legal Q&A pairs minimum
4. **Specialized Models**: Start with models pre-trained on legal text
5. **Instruction Tuning**: Focus on following legal reasoning patterns rather than memorizing facts


## 🛠️ Technical Details

### Model Configurations
```python
# Training Arguments
TrainingArguments(
    num_train_epochs=5,
    per_device_train_batch_size=4,
    learning_rate=5e-5,
    warmup_steps=100,
    weight_decay=0.01,
    eval_strategy="steps",
    save_strategy="steps"
)
```

### Data Processing
```python
# Text preprocessing for legal content
def tokenize_function(examples):
    model_inputs = tokenizer(
        examples["input_text"],
        max_length=512,
        truncation=True,
        padding=False
    )
    # Target sequence processing
    labels = tokenizer(
        examples["target_text"], 
        max_length=256,
        truncation=True
    )
    return model_inputs
```

## 📊 Results Analysis

### Expected vs. Actual Outcomes
This project **intentionally demonstrates the limitations** of fine-tuning small models on specialized domains:

**Question**: "What is the legal standard for establishing proximate cause in tort law?"

- **Original**: "proximate grounds" *(meaningless)*
- **Span Corruption**: "a court case" *(irrelevant)*
- **Q&A Fine-tuned**: "Prohibition of proximate cause in tort law requires proximate cause to be found in tort law." *(circular, incorrect)*

### Analysis
- **No Meaningful Improvement**: All models fail to provide accurate legal information
- **Potential Degradation**: Fine-tuned models may perform worse on general tasks
- **Learning Value**: Demonstrates why this approach is inappropriate for legal applications
- **Technical Success**: Fine-tuning pipeline worked correctly despite poor outputs

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Google**: FLAN-T5 model architecture
- **Hugging Face**: Transformers library and model hosting
- **Cornell Law School**: Legal reference materials
- **Legal Community**: Open access to legal knowledge
