#!/usr/bin/env python3
"""
Automated extraction script for data_contracts.py refactoring.
This script intelligently splits the monolithic file into organized modules.
"""

import re
from pathlib import Path

# Read the backup file
backup_file = Path('src/engine/models/data_contracts_backup.py')
with open(backup_file, 'r') as f:
    content = f.read()
    lines = f.readlines()

# Reset file pointer
with open(backup_file, 'r') as f:
    lines = f.readlines()

# Extract header (imports and module docstring)
header_lines = []
for i, line in enumerate(lines):
    if line.startswith('class ') or line.startswith('def '):
        break
    header_lines.append(line)

header = ''.join(header_lines)

# Categories for classification
enums = []
engagement_models = []
consultant_models = []
analysis_models = []
event_models = []
other_models = []
factories = []
validators = []
transformers = []

# Parse all items
current_item_lines = []
current_item_name = None
in_item = False

for line in lines[len(header_lines):]:
    if line.startswith('class ') or line.startswith('def '):
        # Save previous item
        if current_item_lines:
            item_text = ''.join(current_item_lines)
            # Classify and store
            if 'Enum' in current_item_lines[0]:
                enums.append((current_item_name, item_text))
            elif current_item_name and 'create_' in current_item_name:
                factories.append((current_item_name, item_text))
            elif current_item_name and 'validate' in current_item_name.lower():
                validators.append((current_item_name, item_text))
            elif 'Engagement' in current_item_name or 'Clarification' in current_item_name or 'Exploration' in current_item_name or 'Workflow' in current_item_name or 'Deliverable' in current_item_name or 'FailureMode' in current_item_name:
                engagement_models.append((current_item_name, item_text))
            elif 'Consultant' in current_item_name or 'Scoring' in current_item_name:
                consultant_models.append((current_item_name, item_text))
            elif 'Mental' in current_item_name or 'Reasoning' in current_item_name or 'Research' in current_item_name or 'Cognitive' in current_item_name or 'Context' in current_item_name or 'Hallucination' in current_item_name:
                analysis_models.append((current_item_name, item_text))
            else:
                other_models.append((current_item_name, item_text))

        # Start new item
        current_item_lines = [line]
        match = re.match(r'(?:class|def)\s+(\w+)', line)
        current_item_name = match.group(1) if match else 'Unknown'
        in_item = True
    elif in_item:
        if line and not line[0].isspace() and not line.strip().startswith('#') and line.strip():
            # End of item
            in_item = False
        else:
            current_item_lines.append(line)

# Save last item
if current_item_lines and current_item_name:
    item_text = ''.join(current_item_lines)
    if 'Enum' in current_item_lines[0]:
        enums.append((current_item_name, item_text))
    elif current_item_name and 'create_' in current_item_name:
        factories.append((current_item_name, item_text))
    elif current_item_name and 'validate' in current_item_name.lower():
        validators.append((current_item_name, item_text))
    elif 'Engagement' in current_item_name or 'Clarification' in current_item_name:
        engagement_models.append((current_item_name, item_text))
    elif 'Consultant' in current_item_name or 'Scoring' in current_item_name:
        consultant_models.append((current_item_name, item_text))
    elif 'Mental' in current_item_name or 'Reasoning' in current_item_name or 'Research' in current_item_name:
        analysis_models.append((current_item_name, item_text))
    else:
        other_models.append((current_item_name, item_text))

# Print classification results
print(f"Classification Results:")
print(f"  Enums: {len(enums)}")
print(f"  Engagement Models: {len(engagement_models)}")
print(f"  Consultant Models: {len(consultant_models)}")
print(f"  Analysis Models: {len(analysis_models)}")
print(f"  Other Models: {len(other_models)}")
print(f"  Factories: {len(factories)}")
print(f"  Validators: {len(validators)}")
print(f"  Transformers: {len(transformers)}")

# Create enums.py
enums_content = header + '\n\n'
for name, text in enums:
    enums_content += text + '\n\n'

Path('src/engine/models/data_contracts/models/enums.py').write_text(enums_content)
print(f"\n✅ Created models/enums.py ({len(enums)} enums)")

# Create engagement_models.py
engagement_header = """\"\"\"
Engagement Domain Models

Models related to engagement lifecycle, clarification, exploration,
and workflow management.
\"\"\"

from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
from .enums import *

"""
engagement_content = engagement_header
for name, text in engagement_models:
    engagement_content += text + '\n\n'

Path('src/engine/models/data_contracts/models/engagement_models.py').write_text(engagement_content)
print(f"✅ Created models/engagement_models.py ({len(engagement_models)} models)")

# Create consultant_models.py
consultant_header = """\"\"\"
Consultant Domain Models

Models related to consultant specialization, scoring, and blueprints.
\"\"\"

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from .enums import *

"""
consultant_content = consultant_header
for name, text in consultant_models:
    consultant_content += text + '\n\n'

Path('src/engine/models/data_contracts/models/consultant_models.py').write_text(consultant_content)
print(f"✅ Created models/consultant_models.py ({len(consultant_models)} models)")

# Create analysis_models.py
analysis_header = """\"\"\"
Analysis Domain Models

Models related to mental models, reasoning, research intelligence,
and cognitive state.
\"\"\"

from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field
from .enums import *

"""
analysis_content = analysis_header
for name, text in analysis_models:
    analysis_content += text + '\n\n'

# Add other models too
for name, text in other_models:
    analysis_content += text + '\n\n'

Path('src/engine/models/data_contracts/models/analysis_models.py').write_text(analysis_content)
print(f"✅ Created models/analysis_models.py ({len(analysis_models) + len(other_models)} models)")

# Create factories
if factories:
    factory_header = """\"\"\"
Event Factory Functions

Functions for creating CloudEvents-compliant event structures.
\"\"\"

from typing import Dict, Any
from datetime import datetime, timezone
from uuid import uuid4
from ..models.engagement_models import *
from ..models.consultant_models import *
from ..models.analysis_models import *
from ..models.enums import *

"""
    factory_content = factory_header
    for name, text in factories:
        factory_content += text + '\n\n'

    Path('src/engine/models/data_contracts/factories/event_factory.py').write_text(factory_content)
    print(f"✅ Created factories/event_factory.py ({len(factories)} functions)")

# Create validators
if validators:
    validator_header = """\"\"\"
Contract Validators

Validation functions for data contract compliance.
\"\"\"

from typing import Dict, Any

"""
    validator_content = validator_header
    for name, text in validators:
        validator_content += text + '\n\n'

    Path('src/engine/models/data_contracts/validators/contract_validators.py').write_text(validator_content)
    print(f"✅ Created validators/contract_validators.py ({len(validators)} functions)")

# Create __init__ files
Path('src/engine/models/data_contracts/models/__init__.py').write_text('')
Path('src/engine/models/data_contracts/factories/__init__.py').write_text('')
Path('src/engine/models/data_contracts/validators/__init__.py').write_text('')
Path('src/engine/models/data_contracts/transformers/__init__.py').write_text('')

print("\n✅ Extraction complete!")
print("\nNext step: Create root __init__.py with re-exports")
