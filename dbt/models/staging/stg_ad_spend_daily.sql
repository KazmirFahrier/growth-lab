select
    cast(date as date) as date,
    channel,
    cast(spend as double) as spend,
    cast(impressions as bigint) as impressions,
    cast(clicks as bigint) as clicks
from {{ source('raw', 'ad_spend_daily') }}
