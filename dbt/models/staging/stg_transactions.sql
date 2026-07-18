select
    cast(txn_id as bigint) as txn_id,
    cast(user_id as bigint) as user_id,
    cast(txn_date as date) as txn_date,
    cast(amount as double) as amount,
    cast(is_fraud as boolean) as is_fraud
from {{ source('raw', 'transactions') }}
