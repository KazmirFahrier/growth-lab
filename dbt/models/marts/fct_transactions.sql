select
    t.txn_id,
    t.user_id,
    t.txn_date,
    t.amount,
    t.is_fraud,
    u.channel as acquisition_channel,
    u.plan,
    u.signup_date
from {{ ref('stg_transactions') }} t
join {{ ref('dim_users') }} u using (user_id)
