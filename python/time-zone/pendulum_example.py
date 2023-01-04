#!/usr/bin/env python3
"""Friendship with pytz ended

Managing datetimes with timezone information.
Requires the module `pendulum` from PyPI.
"""
import pendulum
import pendulum.tz


# new datetime object from utc 0
dt = pendulum.datetime(2022, 12, 1)
print(f'new datetime with naive timezone:     {dt}')

# new datetime object with specific time zone
dt = pendulum.datetime(2022, 12, 1, tz='America/Recife')
print(f'new datetime with timezone string:    {dt}')

tz = pendulum.tz.timezone('America/Recife')
dt = pendulum.datetime(2022, 12, 1, tz=tz)
print(f'new datetime with timezone object:    {dt}')

# new datetime object with local timezone
dt = pendulum.local(2022, 12, 1)
print(f'new datetime with local timezone:     {dt}')

# current datetime with local timezone
dt = pendulum.now()
print(f'current datetime with local timezone: {dt}')

# parsing datetime
dt = pendulum.parser.parse('2022-12-01T22:11:00')
print(f'parsing without timezone:             {dt}')

dt = pendulum.parser.parse('2022-12-01T22:11:00-03:00')
print(f'parsing with timezone:                {dt}')
