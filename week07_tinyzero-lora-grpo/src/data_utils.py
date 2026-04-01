"""Countdown dataset + prompt formatting (TinyZero-compatible split)."""                                      
                                                                                                                                
from __future__ import annotations                                                                                             
                                                                                                                                
from pathlib import Path                                                                                                       
                                                          
from datasets import Dataset, DatasetDict, load_dataset                                                                        
                                                          
DEFAULT_DATASET = "Jiayi-Pan/Countdown-Tasks-3to4"                                                                             
# Match TinyZero's exact index-based split (no random seed, no filter)
# countdown.py: train = raw[: TRAIN_SIZE], test = raw[TRAIN_SIZE : TRAIN_SIZE + TEST_SIZE]                                     
TRAIN_SIZE = 327680                                                                                                            
TEST_SIZE = 1024                                                                                                               
                                                                                                                                
                                                                                                                                
def _project_root() -> Path:                                                                                                   
    return Path(__file__).resolve().parents[1]                                                                                 
                                                                                                                                
                                                                                                                                
def default_prompt_template_path() -> Path:                                                                                    
    return _project_root() / "data" / "prompts" / "prompt_template.txt"                                                        
                                                                                                                                
                                                                                                                                
def load_prompt_template(path: Path | None = None) -> str:                                                                     
    p = path or default_prompt_template_path()                                                                                 
    return p.read_text(encoding="utf-8")                                                                                       
                                                                                                                                
                                                                                                                                
def format_countdown_prompt(nums: list, target: int | float, template: str | None = None) -> str:                              
    """Fill template placeholders: {nums_line}, {target}."""                                                                   
    t = template if template is not None else load_prompt_template()                                                           
    nums_line = ", ".join(str(x) for x in nums)                                                                                
    return t.format(nums_line=nums_line, target=target)                                                                        
                                                                                                                                
                                                                                                                                
def build_countdown_dataset() -> DatasetDict:                                                                                                                        
    raw = load_dataset(DEFAULT_DATASET, split="train")                                                                         
    assert len(raw) > TRAIN_SIZE + TEST_SIZE, (                                                                                
        f"Dataset too small: {len(raw)} <= {TRAIN_SIZE + TEST_SIZE}"                                                           
    )                                                                                                                          
    train_dataset = raw.select(range(TRAIN_SIZE))                                                                              
    test_dataset = raw.select(range(TRAIN_SIZE, TRAIN_SIZE + TEST_SIZE))                                                       
    return DatasetDict({"train": train_dataset, "test": test_dataset})                                                         
                                                                                                                                
                                                                                                                                
def add_prompt_column(ds: Dataset, template: str | None = None) -> Dataset:                                                    
    tpl = template if template is not None else load_prompt_template()                                                         
                                                                                                                                
    def _row(ex: dict) -> dict:                                                                                                
        prompt = format_countdown_prompt(ex["nums"], ex["target"], tpl)                                                        
        return {"prompt": prompt}                                                                                              
                                                                                                                                
    return ds.map(_row)                                                                                                        
