#pragma once

#include "host.h"

class HostSilent : public Host
{
  private:
    bool quit;

  public:
    HostSilent();

    virtual void onFrame();
    virtual void onGameBegin(const Info& info);
    virtual void onDealDone(const Info& info);
    virtual void onGameDone(const Info& info);

    virtual bool wantToQuit() const;
    virtual void resetWantToQuit();
};
