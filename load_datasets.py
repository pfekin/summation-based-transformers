def load_dataset_for_lm(dataset_name, tokenizer, max_length=512, subset=None):
    """
    Generalized function to load various text datasets for language modeling.
    
    Args:
        dataset_name: Name of the dataset ('wikitext-2', 'imdb', 'ag_news', 'cmu-book-summaries')
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        subset: Dataset subset/configuration (if applicable)
    
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset) or (train_dataset, test_dataset)
    """
    from datasets import load_dataset
    
    # Dataset configurations
    dataset_configs = {
        'wikitext-2': {
            'path': 'wikitext',
            'name': 'wikitext-2-raw-v1',
            'text_column': 'text',
            'splits': ['train', 'validation', 'test'],
            'has_validation': True
        },
        'imdb': {
            'path': 'imdb',
            'name': None,
            'text_column': 'text',
            'splits': ['train', 'test'],
            'has_validation': False
        },
        'ag_news': {
            'path': 'ag_news',
            'name': None,
            'text_column': 'text',
            'splits': ['train', 'test'],
            'has_validation': False
        },
        'cmu-book-summaries': {
            'path': 'textminr/cmu-book-summaries',
            'name': None,
            'text_column': 'summary',
            'splits': ['train'],
            'has_validation': False
        }
    }
    
    if dataset_name not in dataset_configs:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Supported datasets: {list(dataset_configs.keys())}")
    
    config = dataset_configs[dataset_name]
    
    def tokenize_function(examples):
        # Get the appropriate text column
        text_column = config['text_column']
        
        # Process each text individually to maintain batch consistency
        input_ids = []
        attention_masks = []
        
        for text in examples[text_column]:
            if text and text.strip():
                # Tokenize non-empty text
                tokenized = tokenizer(text, truncation=True, padding=False, max_length=max_length)
                input_ids.append(tokenized['input_ids'])
                attention_masks.append(tokenized['attention_mask'])
            else:
                # For empty texts, add empty lists (will be filtered out later)
                input_ids.append([])
                attention_masks.append([])
        
        return {'input_ids': input_ids, 'attention_mask': attention_masks}
    
    # Load datasets based on configuration
    datasets = {}
    
    if config['has_validation']:
        # Load train, validation, and test splits
        train_dataset = load_dataset(config['path'], config['name'], split='train')
        val_dataset = load_dataset(config['path'], config['name'], split='validation')
        test_dataset = load_dataset(config['path'], config['name'], split='test')
        
        datasets['train'] = train_dataset
        datasets['validation'] = val_dataset
        datasets['test'] = test_dataset
        
    elif len(config['splits']) == 2:
        # Load train and test splits only
        train_dataset = load_dataset(config['path'], config['name'], split='train')
        test_dataset = load_dataset(config['path'], config['name'], split='test')
        
        datasets['train'] = train_dataset
        datasets['test'] = test_dataset
        
    else:
        # Only train split available (like CMU book summaries)
        train_dataset = load_dataset(config['path'], config['name'], split='train')
        datasets['train'] = train_dataset
    
    # Tokenize datasets
    tokenized_datasets = {}
    
    for split_name, dataset in datasets.items():
        # Remove columns that aren't needed (keep only text column initially)
        columns_to_remove = [col for col in dataset.column_names if col != config['text_column']]
        
        tokenized = dataset.map(
            tokenize_function, 
            batched=True, 
            remove_columns=columns_to_remove
        )
        
        # Filter out empty sequences
        tokenized = tokenized.filter(lambda x: len(x['input_ids']) > 1)
        
        # Set format for PyTorch
        tokenized.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        
        tokenized_datasets[split_name] = tokenized
    
    # Return appropriate tuple based on available splits
    if config['has_validation']:
        return tokenized_datasets['train'], tokenized_datasets['validation'], tokenized_datasets['test']
    elif 'test' in tokenized_datasets:
        return tokenized_datasets['train'], tokenized_datasets['test']
    else:
        return tokenized_datasets['train']


# Convenience functions for specific datasets
def load_wikitext2(tokenizer, max_length=512):
    """Load WikiText-2 dataset."""
    return load_dataset_for_lm('wikitext-2', tokenizer, max_length)

def load_imdb(tokenizer, max_length=512):
    """Load IMDB reviews dataset."""
    return load_dataset_for_lm('imdb', tokenizer, max_length)

def load_ag_news(tokenizer, max_length=512):
    """Load AG News dataset."""
    return load_dataset_for_lm('ag_news', tokenizer, max_length)

def load_cmu_book_summaries(tokenizer, max_length=512, split_data=True, val_size=0.1, test_size=0.1):
    """
    Load CMU Book Summaries dataset.
    
    Args:
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        split_data: Whether to split the train data into train/val/test
        val_size: Fraction of data for validation (if split_data=True)
        test_size: Fraction of data for test (if split_data=True)
    
    Returns:
        If split_data=True: (train_data, val_data, test_data)
        If split_data=False: train_data
    """
    train_data = load_dataset_for_lm('cmu-book-summaries', tokenizer, max_length)
    
    if not split_data:
        return train_data
    
    # Calculate split sizes
    total_size = len(train_data)
    test_split_size = int(total_size * test_size)
    val_split_size = int(total_size * val_size)
    train_split_size = total_size - test_split_size - val_split_size
    
    # Split the dataset
    train_val_data = train_data.train_test_split(test_size=test_split_size, shuffle=True, seed=42)
    train_data = train_val_data['train']
    test_data = train_val_data['test']
    
    # Further split train into train and validation
    train_val_split = train_data.train_test_split(test_size=val_split_size/(total_size - test_split_size), shuffle=True, seed=42)
    train_data = train_val_split['train']
    val_data = train_val_split['test']
    
    return train_data, val_data, test_data
