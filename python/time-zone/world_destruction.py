#!/usr/bin/env python3
# Requires pytz modules from PyPI
"""Adding time zone information to datetimes

Speak about destruction
"""
from datetime import datetime
from pytz import timezone


# time-zone naive string
time = '2022-10-18 10:51:1'
fmt_input = '%Y-%m-%d %H:%M:%S'

# output format with time-zone field
fmt_output = '%Y-%m-%d %H:%M:%S %p %Z'

# Parsing string into a naive datetime
dt_input = datetime.strptime(time, fmt_input)

# Specifying a timezone
tz = timezone('America/Los_Angeles')

# Localizing the timezone
dt_output = tz.localize(dt_input)

# Printing both naive and tz-aware datetimes
# with time-zone fields
print(dt_input.strftime(fmt_output))
print(dt_output.strftime(fmt_output))
