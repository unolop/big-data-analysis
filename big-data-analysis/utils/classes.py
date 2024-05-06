## S0(Wake):0, S1:1, S2:2, S3:3, REM:4
def convert3class(y): 
    if y >= 4: ## REM 
        return 2
    ## NON-REM 
    if y == 3: ## DeepSleep 
        return 1
    if y == 2: ## & 1 LightSleep 
        return 1
    else:      # 0 Wake 
        return y

def convert4class(y):
    if y >= 4: ## REM 
        return 3
    ## NON-REM 
    if y == 3: ## DeepSleep 
        return 2
    if y == 2: ## & 1 LightSleep 
        return 1
    else:    # wake = 0 
        return y

def convert5class(y):
    if y >= 4:
        return 4
    else:    
        return y
