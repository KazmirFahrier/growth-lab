select
    cast(user_id as bigint) as user_id,
    cast(signup_date as date) as signup_date,
    channel,
    cast(subscribed as boolean) as subscribed,
    plan
from {{ source('raw', 'signups') }}
