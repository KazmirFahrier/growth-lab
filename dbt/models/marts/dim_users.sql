select
    user_id,
    signup_date,
    channel,
    subscribed as is_paid,
    plan
from {{ ref('stg_signups') }}
